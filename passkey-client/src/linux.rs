//! Multi-device USB HID WebAuthn client for Linux.
//!
//! [`LinuxClient`] wraps a set of [`LinuxAuthenticator`]s, one for each detected FIDO-compatible
//! HID device, and races a WebAuthn ceremony across all of them. Whichever security key the user
//! touches first wins; the rest receive a `CTAPHID_CANCEL` command so their in-flight transactions
//! are cancelled cleanly.
//!
//! Each per-device CTAP request is tailored to that device's capabilities (advertised in its cached
//! `authenticatorGetInfo` response). Algorithms that the device doesn't support are filtered out,
//! resident-key-incapable devices are skipped when an RP requires `rk`, and so on. Only the
//! response from the winning authenticator is returned.
//!
//! ## Usage
//!
//! ```ignore
//! use passkey_client::linux::LinuxClient;
//!
//! let mut client = LinuxClient::open_all().await?;
//! let created = client.register(origin, request, DefaultClientData).await?;
//! ```

use std::sync::Arc;

use coset::{Algorithm, iana::EnumI64};
use passkey_authenticator::linux::{LinuxAuthenticator, OpenError};
use passkey_authenticator::public_key_der_from_cose_key;
use passkey_types::{
    crypto::sha256, ctap2, encoding,
    webauthn::{
        self, AuthenticatedPublicKeyCredential, AuthenticatorAssertionResponse,
        AuthenticatorAttachment, AuthenticatorAttestationResponse, ClientDataType,
        CollectedClientData, CreatedPublicKeyCredential, CredentialCreationOptions,
        CredentialRequestOptions, PublicKeyCredentialParameters, PublicKeyCredentialType,
        ResidentKeyRequirement, UserVerificationRequirement,
    },
};
use serde::Serialize;
use tokio::sync::mpsc;

use crate::{ClientData, Fetcher, Origin, RpIdVerifier, WebauthnError};

/// A WebAuthn client backed by zero or more USB security keys serving as authenticators.
pub struct LinuxClient<P, F>
where
    P: public_suffix::EffectiveTLDProvider + Sync + 'static,
    F: Fetcher + Sync,
{
    devices: Vec<Arc<LinuxAuthenticator>>,
    rp_id_verifier: RpIdVerifier<P, F>,
    uv_when_preferred: bool,
}

impl LinuxClient<public_suffix::PublicSuffixList, ()> {
    /// Build a `LinuxClient` over the supplied authenticators using the default public-suffix list
    /// TLD provider.
    pub fn new(authenticators: Vec<LinuxAuthenticator>) -> Self {
        Self {
            devices: authenticators.into_iter().map(Arc::new).collect(),
            rp_id_verifier: RpIdVerifier::new(public_suffix::DEFAULT_PROVIDER, None),
            uv_when_preferred: true,
        }
    }

    /// Enumerate every FIDO-capable USB HID device on the system and open each
    /// one. Devices that fail to open are skipped silently.
    pub async fn open_all() -> Result<Self, OpenError> {
        let infos = LinuxAuthenticator::list_devices()?;
        let mut authenticators = Vec::new();
        for info in infos {
            if let Ok(auth) = LinuxAuthenticator::open(&info.path).await {
                authenticators.push(auth);
            }
        }
        Ok(Self::new(authenticators))
    }
}

impl<P, F> LinuxClient<P, F>
where
    P: public_suffix::EffectiveTLDProvider + Sync + 'static,
    F: Fetcher + Sync,
{
    /// Build a `LinuxClient` with a custom TLD provider and optional fetcher.
    pub fn new_with_custom_tld_provider(
        authenticators: Vec<LinuxAuthenticator>,
        custom_provider: P,
        fetcher: Option<F>,
    ) -> Self {
        Self {
            devices: authenticators.into_iter().map(Arc::new).collect(),
            rp_id_verifier: RpIdVerifier::new(custom_provider, fetcher),
            uv_when_preferred: true,
        }
    }
}

impl<P, F> LinuxClient<P, F>
where
    P: public_suffix::EffectiveTLDProvider + Sync + 'static,
    F: Fetcher + Sync,
{
    /// Mirror of `Client::user_verification_when_preferred`.
    pub fn user_verification_when_preferred(mut self, enabled: bool) -> Self {
        self.uv_when_preferred = enabled;
        self
    }

    /// How many authenticators this client will dispatch to.
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Register a credential by racing every connected security key.
    pub async fn register<D: ClientData<E>, E: Serialize + Clone>(
        &self,
        origin: impl Into<Origin<'_>>,
        request: CredentialCreationOptions,
        client_data: D,
    ) -> Result<CreatedPublicKeyCredential, WebauthnError> {
        let origin = origin.into();
        let opts = request.public_key;

        let rp_id = self
            .rp_id_verifier
            .assert_domain(&origin, opts.rp.id.as_deref())
            .await?;
        let rp_id = rp_id.to_owned();

        let collected_client_data = CollectedClientData::<E> {
            ty: ClientDataType::Create,
            challenge: encoding::base64url(&opts.challenge),
            origin: origin.to_string(),
            cross_origin: None,
            extra_data: client_data.extra_client_data(),
            unknown_keys: Default::default(),
        };
        let client_data_json = serde_json::to_string(&collected_client_data)
            .map_err(|_| WebauthnError::SerializationError)?;
        let client_data_hash = client_data
            .client_data_hash()
            .unwrap_or_else(|| sha256(client_data_json.as_bytes()).to_vec());

        let pub_key_cred_params = if opts.pub_key_cred_params.is_empty() {
            PublicKeyCredentialParameters::default_algorithms()
        } else {
            opts.pub_key_cred_params
        };

        let uv_requirement = opts
            .authenticator_selection
            .as_ref()
            .map(|s| s.user_verification)
            .unwrap_or_default();
        let uv = self.ctap_uv_option(uv_requirement);

        // Per-device filter + tailored request build.
        let mut candidates: Vec<(Arc<LinuxAuthenticator>, ctap2::make_credential::Request)> =
            Vec::with_capacity(self.devices.len());
        for device in &self.devices {
            let info = device.info();

            let rk = map_rk(opts.authenticator_selection.as_ref(), &info);
            if rk && !info.options.as_ref().is_some_and(|o| o.rk) {
                // RP requires a resident key but this device can't store one.
                continue;
            }
            if uv && !device_supports_uv(&info) {
                // RP requires UV but this device has no built-in UV (and we don't implement
                // clientPIN yet, so PIN-only devices can't satisfy `uv`).
                continue;
            }

            let matched_params = filter_algorithms(&pub_key_cred_params, &info);
            if matched_params.is_empty() {
                // None of the algorithms requested by the RP are supported by this authenticator.
                continue;
            }

            let request = ctap2::make_credential::Request {
                client_data_hash: client_data_hash.clone().into(),
                rp: ctap2::make_credential::PublicKeyCredentialRpEntity {
                    id: rp_id.clone(),
                    name: Some(opts.rp.name.clone()),
                },
                user: opts.user.clone(),
                pub_key_cred_params: matched_params,
                exclude_list: opts.exclude_credentials.clone(),
                extensions: None,
                options: ctap2::make_credential::Options { rk, up: false, uv },
                pin_auth: None,
                pin_protocol: None,
            };
            candidates.push((Arc::clone(device), request));
        }

        if candidates.is_empty() {
            return Err(WebauthnError::NotSupportedError);
        }

        let ctap2_response = race_make_credential(candidates).await?;

        let credential_id = ctap2_response
            .auth_data
            .attested_credential_data
            .as_ref()
            .ok_or_else(|| WebauthnError::AuthenticatorError(0x7F))?;
        let alg = match credential_id.key.alg.as_ref() {
            Some(Algorithm::PrivateUse(val)) => *val,
            Some(Algorithm::Assigned(a)) => EnumI64::to_i64(a),
            // In the case that the algorithm is unknown, default to 0 (Reserved)
            _ => 0,
        };
        let public_key = Some(
            public_key_der_from_cose_key(&credential_id.key)
                .map_err(|e| WebauthnError::AuthenticatorError(e.into()))?,
        );
        let attestation_object = ctap2_response.as_webauthn_bytes();

        Ok(CreatedPublicKeyCredential {
            id: encoding::base64url(credential_id.credential_id()),
            raw_id: credential_id.credential_id().to_vec().into(),
            ty: PublicKeyCredentialType::PublicKey,
            response: AuthenticatorAttestationResponse {
                client_data_json: Vec::from(client_data_json).into(),
                authenticator_data: ctap2_response.auth_data.to_vec().into(),
                public_key,
                public_key_algorithm: alg,
                attestation_object,
                // Every `LinuxAuthenticator` uses the Usb transport.
                transports: Some(vec![webauthn::AuthenticatorTransport::Usb]),
            },
            authenticator_attachment: Some(AuthenticatorAttachment::CrossPlatform),
            client_extension_results: Default::default(),
        })
    }

    /// Get assertion for a credential by racing every connected security key.
    pub async fn authenticate<D: ClientData<E>, E: Serialize + Clone>(
        &self,
        origin: impl Into<Origin<'_>>,
        request: CredentialRequestOptions,
        client_data: D,
    ) -> Result<AuthenticatedPublicKeyCredential, WebauthnError> {
        let origin = origin.into();
        let opts = request.public_key;

        let rp_id = self
            .rp_id_verifier
            .assert_domain(&origin, opts.rp_id.as_deref())
            .await?;
        let rp_id = rp_id.to_owned();

        let collected_client_data = CollectedClientData::<E> {
            ty: ClientDataType::Get,
            challenge: encoding::base64url(&opts.challenge),
            origin: origin.to_string(),
            cross_origin: None,
            extra_data: client_data.extra_client_data(),
            unknown_keys: Default::default(),
        };
        let client_data_json = serde_json::to_string(&collected_client_data)
            .map_err(|_| WebauthnError::SerializationError)?;
        let client_data_hash = client_data
            .client_data_hash()
            .unwrap_or_else(|| sha256(client_data_json.as_bytes()).to_vec());

        let uv = self.ctap_uv_option(opts.user_verification);

        let mut candidates: Vec<(Arc<LinuxAuthenticator>, ctap2::get_assertion::Request)> =
            Vec::with_capacity(self.devices.len());
        for device in &self.devices {
            let info = device.info();
            if uv && !device_supports_uv(&info) {
                continue;
            }
            let request = ctap2::get_assertion::Request {
                rp_id: rp_id.clone(),
                client_data_hash: client_data_hash.clone().into(),
                allow_list: opts.allow_credentials.clone(),
                extensions: None,
                options: ctap2::get_assertion::Options {
                    up: true,
                    uv,
                },
                pin_auth: None,
                pin_protocol: None,
            };
            candidates.push((Arc::clone(device), request));
        }

        if candidates.is_empty() {
            return Err(WebauthnError::NotSupportedError);
        }

        let ctap2_response = race_get_assertion(candidates).await?;

        let credential_id_bytes = match ctap2_response.credential {
            Some(c) => c.id.to_vec(),
            None => return Err(WebauthnError::CredentialNotFound),
        };
        Ok(AuthenticatedPublicKeyCredential {
            id: encoding::base64url(&credential_id_bytes),
            raw_id: credential_id_bytes.into(),
            ty: PublicKeyCredentialType::PublicKey,
            response: AuthenticatorAssertionResponse {
                client_data_json: Vec::from(client_data_json).into(),
                authenticator_data: ctap2_response.auth_data.to_vec().into(),
                signature: ctap2_response.signature,
                user_handle: ctap2_response.user.map(|user| user.id),
                attestation_object: None,
            },
            authenticator_attachment: Some(AuthenticatorAttachment::CrossPlatform),
            client_extension_results: Default::default(),
        })
    }

    fn ctap_uv_option(&self, requirement: UserVerificationRequirement) -> bool {
        match requirement {
            UserVerificationRequirement::Discouraged => false,
            UserVerificationRequirement::Required => true,
            UserVerificationRequirement::Preferred => self.uv_when_preferred,
        }
    }
}

/// Whether the given `get_info::Response` indicates that the device supports a user verification
/// method that's already configured.
fn device_supports_uv(info: &ctap2::get_info::Response) -> bool {
    info.options
        .as_ref()
        .and_then(|o| o.uv)
        .unwrap_or(false)
}

/// Copy of `Client::map_rk`.
// TODO: Should that method be moved out of `Client`? It's pretty much an exact duplicate.
fn map_rk(
    sel: Option<&webauthn::AuthenticatorSelectionCriteria>,
    info: &ctap2::get_info::Response,
) -> bool {
    let supports_rk = info.options.as_ref().is_some_and(|o| o.rk);
    match sel.unwrap_or(&Default::default()) {
        webauthn::AuthenticatorSelectionCriteria {
            resident_key: Some(ResidentKeyRequirement::Required),
            ..
        } => true,
        webauthn::AuthenticatorSelectionCriteria {
            resident_key: Some(ResidentKeyRequirement::Preferred),
            ..
        } => supports_rk,
        webauthn::AuthenticatorSelectionCriteria {
            resident_key: Some(ResidentKeyRequirement::Discouraged),
            ..
        } => false,
        webauthn::AuthenticatorSelectionCriteria {
            resident_key: None,
            require_resident_key,
            ..
        } => *require_resident_key,
    }
}

/// Keep only the algorithms the device's `get_info` advertises (if it advertises any). The
/// algorithms field is optional, so if no algorithms are advertised then just return the request
/// unmodified.
fn filter_algorithms(
    requested: &[PublicKeyCredentialParameters],
    info: &ctap2::get_info::Response,
) -> Vec<PublicKeyCredentialParameters> {
    let Some(device_algs) = info.algorithms.as_ref() else {
        return requested.to_vec();
    };

    requested
        .iter()
        .filter(|p| device_algs.iter().any(|d| d.alg == p.alg))
        .copied()
        .collect()
}

/// Race `make_credential` across all candidates and cancel the losers once a
/// winner has been found.
async fn race_make_credential(
    candidates: Vec<(Arc<LinuxAuthenticator>, ctap2::make_credential::Request)>,
) -> Result<ctap2::make_credential::Response, WebauthnError> {
    let (tx, mut rx) = mpsc::channel(candidates.len().max(1));
    let auths: Vec<Arc<LinuxAuthenticator>> = candidates
        .iter()
        .map(|(a, _)| Arc::clone(a))
        .collect();

    for (idx, (auth, request)) in candidates.into_iter().enumerate() {
        let tx = tx.clone();
        tokio::spawn(async move {
            let result = auth.make_credential(request).await;
            // Channel may be closed if a winner already emerged — ignore.
            let _ = tx.send((idx, result)).await;
        });
    }
    drop(tx);

    let mut winner_idx: Option<usize> = None;
    let mut winner_response: Option<ctap2::make_credential::Response> = None;
    let mut last_error: Option<WebauthnError> = None;
    while let Some((idx, result)) = rx.recv().await {
        match result {
            Ok(resp) => {
                winner_idx = Some(idx);
                winner_response = Some(resp);
                break;
            }
            Err(sc) => last_error = Some(WebauthnError::from(sc)),
        }
    }

    if let Some(winner_idx) = winner_idx {
        cancel_losers(&auths, winner_idx).await;
    }

    winner_response.ok_or(last_error.unwrap_or(WebauthnError::NotSupportedError))
}

/// Race `get_assertion` across all candidates, similar to `race_make_credential`.
async fn race_get_assertion(
    candidates: Vec<(Arc<LinuxAuthenticator>, ctap2::get_assertion::Request)>,
) -> Result<ctap2::get_assertion::Response, WebauthnError> {
    let (tx, mut rx) = mpsc::channel(candidates.len().max(1));
    let auths: Vec<Arc<LinuxAuthenticator>> = candidates
        .iter()
        .map(|(a, _)| Arc::clone(a))
        .collect();

    for (idx, (auth, request)) in candidates.into_iter().enumerate() {
        let tx = tx.clone();
        tokio::spawn(async move {
            let result = auth.get_assertion(request).await;
            let _ = tx.send((idx, result)).await;
        });
    }
    drop(tx);

    let mut winner_idx: Option<usize> = None;
    let mut winner_response: Option<ctap2::get_assertion::Response> = None;
    let mut all_no_credentials = true;
    let mut last_other_error: Option<WebauthnError> = None;
    while let Some((idx, result)) = rx.recv().await {
        match result {
            Ok(resp) => {
                winner_idx = Some(idx);
                winner_response = Some(resp);
                break;
            }
            Err(sc) => {
                let err = WebauthnError::from(sc);
                if !matches!(err, WebauthnError::CredentialNotFound) {
                    all_no_credentials = false;
                    last_other_error = Some(err);
                }
            }
        }
    }

    if let Some(winner_idx) = winner_idx {
        cancel_losers(&auths, winner_idx).await;
    }

    if let Some(resp) = winner_response {
        Ok(resp)
    } else if all_no_credentials {
        Err(WebauthnError::CredentialNotFound)
    } else {
        Err(last_other_error.unwrap_or(WebauthnError::NotSupportedError))
    }
}

async fn cancel_losers(auths: &[Arc<LinuxAuthenticator>], winner_idx: usize) {
    for (i, auth) in auths.iter().enumerate() {
        if i != winner_idx {
            let _ = auth.cancel().await;
        }
    }
}
