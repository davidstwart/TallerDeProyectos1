import API from "../api/axios";

// ======================================
// LOGIN
// ======================================

export const loginRequest =
  async (data) => {

    const response =
      await API.post(
        "/auth/login",
        data
      );

    return response.data;
  };

// ======================================
// RECOVER PASSWORD
// ======================================

export const recoverPasswordRequest =
  async (data) => {

    const response =
      await API.post(
        "/auth/recover",
        data
      );

    return response.data;
  };

// ======================================
// RESET PASSWORD
// ======================================

export const resetPasswordRequest =
  async (data) => {

    const response =
      await API.post(
        "/auth/reset",
        data
      );

    return response.data;
  };