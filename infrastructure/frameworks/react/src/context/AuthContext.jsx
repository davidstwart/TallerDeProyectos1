import {
  createContext,
  useContext,
  useEffect,
  useState,
} from "react";

const AuthContext =
  createContext(null);

export function AuthProvider({
  children,
}) {

  // =====================================
  // STATES
  // =====================================

  const [user, setUser] =
    useState(null);

  const [loading, setLoading] =
    useState(true);

  // =====================================
  // LOAD SESSION
  // =====================================

  useEffect(() => {

    const token =
      localStorage.getItem(
        "token"
      );

    const savedUser =
      localStorage.getItem(
        "user"
      );

    if (
      token &&
      savedUser
    ) {

      // eslint-disable-next-line react-hooks/set-state-in-effect
      setUser(
        JSON.parse(savedUser)
      );
    }

    setLoading(false);

  }, []);

  // =====================================
  // LOGIN
  // =====================================

  const login = (
    token,
    userData
  ) => {

    localStorage.setItem(
      "token",
      token
    );

    localStorage.setItem(
      "user",
      JSON.stringify(userData)
    );

    setUser(userData);
  };

  // =====================================
  // LOGOUT
  // =====================================

  const logout = () => {

    localStorage.removeItem(
      "token"
    );

    localStorage.removeItem(
      "user"
    );

    setUser(null);
  };

  // =====================================
  // PROVIDER
  // =====================================

  return (

    <AuthContext.Provider
      value={{

        user,
        loading,
        login,
        logout,

        isAuthenticated:
          !!user,
      }}
    >

      {children}

    </AuthContext.Provider>
  );
}

// =====================================
// HOOK
// =====================================

// eslint-disable-next-line react-refresh/only-export-components
export function useAuth() {

  const context =
    useContext(AuthContext);

  if (!context) {

    throw new Error(
      "useAuth debe usarse dentro de AuthProvider"
    );
  }

  return context;
}