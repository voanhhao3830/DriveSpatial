"use client";

import { useState, useEffect } from "react";

export default function LoginPage() {
  const [loginUsername, setLoginUsername] = useState<string>("");
  const [loginPassword, setLoginPassword] = useState<string>("");
  const [loginError, setLoginError] = useState<string>("");

  // Auto-login: check localStorage for saved auth
  useEffect(() => {
    const storedAuth = localStorage.getItem("drivespatial_auth");
    if (storedAuth) {
      try {
        const { username: storedUsername, timestamp } = JSON.parse(storedAuth);
        // Check if auth is less than 7 days old
        const sevenDaysInMs = 7 * 24 * 60 * 60 * 1000;
        if (Date.now() - timestamp < sevenDaysInMs && storedUsername) {
          console.log(`Auto-logged in as: ${storedUsername}`);
          // Redirect to mode selection
          window.location.href = "/mode-select";
        } else {
          // Clear expired auth
          localStorage.removeItem("drivespatial_auth");
        }
      } catch (err) {
        console.error("Failed to parse stored auth:", err);
        localStorage.removeItem("drivespatial_auth");
      }
    }
  }, []);

  /**
   * Handle login form submission
   */
  const handleLogin = (e: React.FormEvent): void => {
    e.preventDefault();
    setLoginError("");

    const requiredPassword = process.env.NEXT_PUBLIC_LOGIN_PASSWORD;

    if (!loginUsername.trim()) {
      setLoginError("Username is required");
      return;
    }

    if (!loginPassword) {
      setLoginError("Password is required");
      return;
    }

    if (loginPassword !== requiredPassword) {
      setLoginError("Invalid password");
      return;
    }

    // Login successful
    const trimmedUsername = loginUsername.trim();
    
    // Store auth in localStorage
    localStorage.setItem("drivespatial_auth", JSON.stringify({
      username: trimmedUsername,
      timestamp: Date.now()
    }));
    
    console.log(`User logged in: ${trimmedUsername}`);
    
    // Redirect to mode selection
    window.location.href = "/mode-select";
  };

  return (
    <div className="h-screen w-screen bg-gradient-to-br from-zinc-50 to-zinc-100 dark:from-zinc-900 dark:to-black font-sans flex items-center justify-center">
      <div className="w-full max-w-md p-8 bg-white dark:bg-zinc-800 rounded-2xl shadow-2xl">
        {/* Logo/Title */}
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-zinc-900 dark:text-zinc-50 mb-2">
            DriveSpatial
          </h1>
          <p className="text-sm text-zinc-600 dark:text-zinc-400">
            AV Data Verification System
          </p>
        </div>

        {/* Login Form */}
        <form onSubmit={handleLogin} className="space-y-6">
          {/* Username Field */}
          <div>
            <label 
              htmlFor="username" 
              className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-2"
            >
              Username
            </label>
            <input
              id="username"
              type="text"
              value={loginUsername}
              onChange={(e) => setLoginUsername(e.target.value)}
              placeholder="Enter your username"
              className="w-full px-4 py-3 bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-700 rounded-lg text-zinc-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all"
              autoComplete="username"
            />
          </div>

          {/* Password Field */}
          <div>
            <label 
              htmlFor="password" 
              className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-2"
            >
              Password
            </label>
            <input
              id="password"
              type="password"
              value={loginPassword}
              onChange={(e) => setLoginPassword(e.target.value)}
              placeholder="Enter your password"
              className="w-full px-4 py-3 bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-700 rounded-lg text-zinc-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all"
              autoComplete="current-password"
            />
          </div>

          {/* Error Message */}
          {loginError && (
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-3">
              <p className="text-sm text-red-600 dark:text-red-400">{loginError}</p>
            </div>
          )}

          {/* Submit Button */}
          <button
            type="submit"
            className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white font-semibold rounded-lg transition-all shadow-lg hover:shadow-xl"
          >
            Login
          </button>
        </form>

      </div>
    </div>
  );
}