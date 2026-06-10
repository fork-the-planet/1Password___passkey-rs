//! Linux USB security key authenticator.
//!
//! [`LinuxAuthenticator`] adapts the CTAPHID transport in
//! [`passkey_transports::hidraw`] to the [`Ctap2Api`](crate::Ctap2Api) trait, so a
//! USB hardware key can be plugged into anything that today drives the in-process
//! [`Authenticator`](crate::Authenticator).
//!
//! ## Usage
//!
//! ```ignore
//! use passkey_authenticator::linux::LinuxAuthenticator;
//!
//! let devices = LinuxAuthenticator::list_devices()?;
//! let mut auth = LinuxAuthenticator::open(&devices[0].path).await?;
//! let info = auth.get_info().await;
//! ```

use std::io;
use std::path::Path;

use passkey_transports::hid::{Command, Message};
use passkey_transports::hidraw::{DeviceInfo, HidDevice, HidrawError, enumerate_fido_devices};
use passkey_types::ctap2::{
    Ctap2Error, StatusCode, U2FError, get_assertion, get_info, make_credential,
};
use tokio::sync::Mutex;

use crate::Ctap2Api;

// Re-export so callers don't need a direct dep on passkey-transports.
pub use passkey_transports::hidraw::{Capabilities, DeviceInfo as HidDeviceInfo, InitResponse};

/// CTAP2 command byte for `authenticatorMakeCredential`.
const CTAP_CMD_MAKE_CREDENTIAL: u8 = 0x01;
/// CTAP2 command byte for `authenticatorGetAssertion`.
const CTAP_CMD_GET_ASSERTION: u8 = 0x02;
/// CTAP2 command byte for `authenticatorGetInfo`.
const CTAP_CMD_GET_INFO: u8 = 0x04;

/// Errors that can occur while constructing a [`LinuxAuthenticator`].
#[derive(Debug)]
#[non_exhaustive]
pub enum OpenError {
    /// The underlying HIDRAW transport returned an error (open / init / I/O).
    Transport(HidrawError),
    /// `CTAPHID_INIT` succeeded but the device does not advertise CTAP2 (CBOR) support.
    NotCtap2,
    /// The device responded to `authenticatorGetInfo` with a CTAP2 status code.
    GetInfo(StatusCode),
    /// The device's `authenticatorGetInfo` response could not be parsed as CBOR.
    InvalidGetInfo,
}

impl From<HidrawError> for OpenError {
    fn from(e: HidrawError) -> Self {
        Self::Transport(e)
    }
}

impl From<io::Error> for OpenError {
    fn from(e: io::Error) -> Self {
        Self::Transport(HidrawError::from(e))
    }
}

impl std::fmt::Display for OpenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenError::Transport(e) => write!(f, "HID transport error: {e}"),
            OpenError::NotCtap2 => f.write_str("device does not advertise CTAP2 support"),
            OpenError::GetInfo(s) => write!(f, "authenticatorGetInfo failed: {s:?}"),
            OpenError::InvalidGetInfo => {
                f.write_str("could not parse authenticatorGetInfo response")
            }
        }
    }
}

impl std::error::Error for OpenError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            OpenError::Transport(e) => Some(e),
            _ => None,
        }
    }
}

/// A CTAP2 authenticator backed by a USB security key reached over Linux HIDRAW.
///
/// Construct with [`LinuxAuthenticator::open`]; enumerate candidate devices with
/// [`LinuxAuthenticator::list_devices`].
pub struct LinuxAuthenticator {
    device: HidDevice,
    channel: u32,
    capabilities: Capabilities,
    /// Cached `authenticatorGetInfo` response, stored as raw CBOR.
    ///
    /// The `Ctap2Api::get_info` trait method returns an owned `Box<Response>` and
    /// takes `&self`, so we can't lazily call the device on every invocation
    /// without interior mutability. Caching the bytes lets us hand out fresh
    /// `Response` values cheaply, and avoids requiring `Clone` on the response
    /// type, which lives in `passkey-types`.
    get_info_cbor: Vec<u8>,
    /// A caller performing a multi-packet CBOR transaction on this device must acquire this lock so
    /// concurrent callers don't interleave packets on the wire.
    txn_lock: Mutex<()>,
}

impl LinuxAuthenticator {
    /// Enumerate FIDO-capable USB HID devices visible on the system.
    pub fn list_devices() -> io::Result<Vec<DeviceInfo>> {
        enumerate_fido_devices()
    }

    /// Open a specific `/dev/hidrawN` path, run `CTAPHID_INIT` to obtain a private
    /// channel, and prime the cached `authenticatorGetInfo` response.
    pub async fn open(path: &Path) -> Result<Self, OpenError> {
        let device = HidDevice::open(path)?;
        let init = device.init().await?;
        if !init.capabilities.supports_cbor() {
            return Err(OpenError::NotCtap2);
        }

        // Fetch authenticatorGetInfo so we can cache it and surface any obvious
        // device-side errors before returning to the caller.
        let raw = send_cbor(&device, init.channel, CTAP_CMD_GET_INFO, &[]).await?;
        // Validate that it parses.
        let _: get_info::Response =
            ciborium::de::from_reader(raw.as_slice()).map_err(|_| OpenError::InvalidGetInfo)?;

        Ok(Self {
            device,
            channel: init.channel,
            capabilities: init.capabilities,
            get_info_cbor: raw,
            txn_lock: Mutex::new(()),
        })
    }

    /// Capabilities reported by the device in its `CTAPHID_INIT` response.
    pub fn capabilities(&self) -> Capabilities {
        self.capabilities
    }

    /// Issue `authenticatorMakeCredential` against the device. Holds an internal mutex for the
    /// duration of the transaction to keep wire packets in order.
    pub async fn make_credential(
        &self,
        request: make_credential::Request,
    ) -> Result<make_credential::Response, StatusCode> {
        let mut body = Vec::new();
        ciborium::ser::into_writer(&request, &mut body)
            .map_err(|_| StatusCode::from(U2FError::Other))?;
        let _guard = self.txn_lock.lock().await;
        let response = send_cbor(&self.device, self.channel, CTAP_CMD_MAKE_CREDENTIAL, &body)
            .await
            .map_err(StatusCode::from)?;
        ciborium::de::from_reader(response.as_slice())
            .map_err(|_| StatusCode::from(Ctap2Error::InvalidCbor))
    }

    /// Issue `authenticatorGetAssertion` against the device. See [`Self::make_credential`].
    pub async fn get_assertion(
        &self,
        request: get_assertion::Request,
    ) -> Result<get_assertion::Response, StatusCode> {
        let mut body = Vec::new();
        ciborium::ser::into_writer(&request, &mut body)
            .map_err(|_| StatusCode::from(U2FError::Other))?;
        let _guard = self.txn_lock.lock().await;
        let response = send_cbor(&self.device, self.channel, CTAP_CMD_GET_ASSERTION, &body)
            .await
            .map_err(StatusCode::from)?;
        ciborium::de::from_reader(response.as_slice())
            .map_err(|_| StatusCode::from(Ctap2Error::InvalidCbor))
    }

    /// Send `CTAPHID_CANCEL` on this authenticator's channel without taking the
    /// transaction mutex.
    ///
    /// This causes any in-flight `CTAPHID_CBOR` request on the same channel to be aborted; the
    /// awaiting `make_credential` / `get_assertion` future will return
    /// `Ctap2Error::KeepAliveCancel`. Calling this on a channel with no in-flight request is a
    /// no-op (the device ignores it).
    pub async fn cancel(&self) -> Result<(), HidrawError> {
        let msg = Message::new(self.channel, Command::Cancel, &[])
            .map_err(|_| HidrawError::MessageTooLarge)?;
        self.device.send(&msg).await
    }

    /// Read and decode the cached `authenticatorGetInfo` response.
    pub fn info(&self) -> get_info::Response {
        ciborium::de::from_reader(self.get_info_cbor.as_slice()).unwrap_or_default()
    }
}

/// Internal error type for CBOR transactions. Maps cleanly to both [`StatusCode`]
/// (for the [`Ctap2Api`] surface) and [`OpenError`] (for the constructor).
#[derive(Debug)]
enum TransactionError {
    Hid(HidrawError),
    Status(StatusCode),
}

impl From<TransactionError> for OpenError {
    fn from(e: TransactionError) -> Self {
        match e {
            TransactionError::Hid(e) => OpenError::Transport(e),
            TransactionError::Status(s) => OpenError::GetInfo(s),
        }
    }
}

impl From<TransactionError> for StatusCode {
    fn from(e: TransactionError) -> Self {
        match e {
            TransactionError::Status(s) => s,
            // CTAP doesn't have a dedicated "transport failed" status code; surface
            // it as the catch-all CTAP1 `U2FError::Other` (0x7F).
            TransactionError::Hid(_) => StatusCode::from(U2FError::Other),
        }
    }
}

/// Run one CTAPHID_CBOR transaction and return the CBOR body of the response.
///
/// Lifted out of [`LinuxAuthenticator`] so it can also be used during construction
/// before `self` exists.
async fn send_cbor(
    device: &HidDevice,
    channel: u32,
    command: u8,
    body: &[u8],
) -> Result<Vec<u8>, TransactionError> {
    let mut payload = Vec::with_capacity(1 + body.len());
    payload.push(command);
    payload.extend_from_slice(body);

    let msg = Message::new(channel, Command::Cbor, &payload)
        .map_err(|_| TransactionError::Hid(HidrawError::MessageTooLarge))?;
    device.send(&msg).await.map_err(TransactionError::Hid)?;

    let response = device.recv(channel).await.map_err(TransactionError::Hid)?;
    if !matches!(response.command, Command::Cbor) {
        return Err(TransactionError::Hid(HidrawError::Protocol(
            "unexpected CTAPHID command in response",
        )));
    }
    let mut bytes = response.payload;
    if bytes.is_empty() {
        return Err(TransactionError::Hid(HidrawError::Protocol(
            "empty CTAPHID_CBOR response",
        )));
    }
    let status = bytes.remove(0);
    if status != 0 {
        return Err(TransactionError::Status(StatusCode::from(status)));
    }
    Ok(bytes)
}

#[async_trait::async_trait]
impl Ctap2Api for LinuxAuthenticator {
    async fn get_info(&self) -> Box<get_info::Response> {
        Box::new(self.info())
    }

    async fn make_credential(
        &mut self,
        request: make_credential::Request,
    ) -> Result<make_credential::Response, StatusCode> {
        LinuxAuthenticator::make_credential(self, request).await
    }

    async fn get_assertion(
        &mut self,
        request: get_assertion::Request,
    ) -> Result<get_assertion::Response, StatusCode> {
        LinuxAuthenticator::get_assertion(self, request).await
    }
}
