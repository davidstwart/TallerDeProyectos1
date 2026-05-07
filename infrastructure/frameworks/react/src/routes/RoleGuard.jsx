import {
  Navigate,
} from "react-router-dom";

import {
  useAuth,
} from "../context/AuthContext";

function RoleGuard({

  children,
  allowedRoles = [],

}) {

  const {
    user,
  } = useAuth();

  // =====================================
  // NO AUTH
  // =====================================

  if (!user) {

    return (
      <Navigate to="/login" />
    );
  }

  // =====================================
  // INVALID ROLE
  // =====================================

  if (
    !allowedRoles.includes(
      user.id_rol
    )
  ) {

    return (
      <Navigate to="/" />
    );
  }

  // =====================================
  // VALID ROLE
  // =====================================

  return children;
}

export default RoleGuard;