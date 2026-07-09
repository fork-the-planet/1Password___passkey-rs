/// Authenticator commands enumerated by the CTAP 2 specification.
#[repr(u8)]
pub enum Ctap2Command {
    /// CTAP_CMD_MAKE_CREDENTIAL
    MakeCredential = 0x01,
    /// CTAP_CMD_GET_ASSERTION
    GetAssertion = 0x02,
    /// CTAP_CMD_GET_INFO
    GetInfo = 0x04,
}
