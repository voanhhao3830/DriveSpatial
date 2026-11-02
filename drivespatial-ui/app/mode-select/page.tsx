"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";

export default function ModeSelection() {
  const router = useRouter();
  const [selectedMode, setSelectedMode] = useState<"graph" | "question" | "evaluation" | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);

  // Check authentication on mount
  useEffect(() => {
    const storedAuth = localStorage.getItem("drivespatial_auth");
    if (storedAuth) {
      try {
        const { username, timestamp } = JSON.parse(storedAuth);
        const sevenDaysInMs = 7 * 24 * 60 * 60 * 1000;
        if (Date.now() - timestamp < sevenDaysInMs && username) {
          setIsAuthenticated(true);
          setLoading(false);
        } else {
          // Auth expired, redirect to login
          localStorage.removeItem("drivespatial_auth");
          router.push("/");
        }
      } catch (err) {
        console.error("Failed to parse stored auth:", err);
        localStorage.removeItem("drivespatial_auth");
        router.push("/");
      }
    } else {
      // No auth, redirect to login
      router.push("/");
    }
  }, [router]);

  const handleModeSelect = (mode: "graph" | "question" | "evaluation") => {
    setSelectedMode(mode);
    // Navigate to the selected mode
    if (mode === "graph") {
      router.push("/graph");
    } else if (mode === "question") {
      router.push("/question");
    } else {
      router.push("/evaluation");
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("drivespatial_auth");
    router.push("/");
  };

  // Show loading state while checking authentication
  if (loading) {
    return (
      <div className="h-screen w-screen bg-gradient-to-br from-zinc-50 to-zinc-100 dark:from-zinc-900 dark:to-black font-sans flex items-center justify-center">
        <div className="text-xl text-zinc-600 dark:text-zinc-400">Loading...</div>
      </div>
    );
  }

  // Only show mode selection if authenticated
  if (!isAuthenticated) {
    return null;
  }

  return (
    <div className="h-screen w-screen bg-gradient-to-br from-zinc-50 to-zinc-100 dark:from-zinc-900 dark:to-black font-sans flex items-center justify-center">
      <div className="w-full max-w-4xl p-8">
        {/* Title */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-zinc-900 dark:text-zinc-50 mb-3">
            DriveSpatial
          </h1>
          <p className="text-lg text-zinc-600 dark:text-zinc-400">
            Select Annotation Mode
          </p>
        </div>

        {/* Mode Selection Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Graph/Annotation Mode */}
          <button
            onClick={() => handleModeSelect("graph")}
            className={`p-8 rounded-2xl shadow-xl transition-all transform hover:scale-105 ${
              selectedMode === "graph"
                ? "bg-gradient-to-br from-blue-500 to-cyan-500 text-white ring-4 ring-blue-300"
                : "bg-white dark:bg-zinc-800 hover:shadow-2xl"
            }`}
          >
            <div className="flex flex-col items-center">
              {/* Icon */}
              <div className="mb-4">
                <svg
                  className={`w-20 h-20 ${
                    selectedMode === "graph"
                      ? "text-white"
                      : "text-blue-500"
                  }`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                  />
                </svg>
              </div>

              {/* Title */}
              <h2
                className={`text-2xl font-bold mb-2 ${
                  selectedMode === "graph"
                    ? "text-white"
                    : "text-zinc-900 dark:text-zinc-50"
                }`}
              >
                Graph Annotation
              </h2>

              {/* Description */}
              <p
                className={`text-sm ${
                  selectedMode === "graph"
                    ? "text-white/90"
                    : "text-zinc-600 dark:text-zinc-400"
                }`}
              >
                Annotate spatial relations, interactions, and occlusions in 3D scenes
              </p>

              {/* Features */}
              <div className="mt-4 space-y-2 text-sm">
                <div
                  className={`flex items-center gap-2 ${
                    selectedMode === "graph"
                      ? "text-white/80"
                      : "text-zinc-500 dark:text-zinc-500"
                  }`}
                >
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path
                      fillRule="evenodd"
                      d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                      clipRule="evenodd"
                    />
                  </svg>
                  <span>3D Visualization</span>
                </div>
                <div
                  className={`flex items-center gap-2 ${
                    selectedMode === "graph"
                      ? "text-white/80"
                      : "text-zinc-500 dark:text-zinc-500"
                  }`}
                >
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path
                      fillRule="evenodd"
                      d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                      clipRule="evenodd"
                    />
                  </svg>
                  <span>Multi-camera Views</span>
                </div>
                <div
                  className={`flex items-center gap-2 ${
                    selectedMode === "graph"
                      ? "text-white/80"
                      : "text-zinc-500 dark:text-zinc-500"
                  }`}
                >
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path
                      fillRule="evenodd"
                      d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                      clipRule="evenodd"
                    />
                  </svg>
                  <span>LiDAR Point Cloud</span>
                </div>
              </div>
            </div>
          </button>

          {/* Question Mode */}
          <button
            onClick={() => handleModeSelect("question")}
            className={`p-8 rounded-2xl shadow-xl transition-all transform hover:scale-105 ${
              selectedMode === "question"
                ? "bg-gradient-to-br from-purple-500 to-pink-500 text-white ring-4 ring-purple-300"
                : "bg-white dark:bg-zinc-800 hover:shadow-2xl"
            }`}
          >
            <div className="flex flex-col items-center">
              {/* Icon */}
              <div className="mb-4">
                <svg
                  className={`w-20 h-20 ${
                    selectedMode === "question"
                      ? "text-white"
                      : "text-purple-500"
                  }`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
              </div>

              {/* Title */}
              <h2
                className={`text-2xl font-bold mb-2 ${
                  selectedMode === "question"
                    ? "text-white"
                    : "text-zinc-900 dark:text-zinc-50"
                }`}
              >
                Question Answering
              </h2>

              {/* Description */}
              <p
                className={`text-sm ${
                  selectedMode === "question"
                    ? "text-white/90"
                    : "text-zinc-600 dark:text-zinc-400"
                }`}
              >
                Answer scene understanding questions with multiple choice options
              </p>

              {/* Features */}
              <div className="mt-4 space-y-2 text-sm">
                <div
                  className={`flex items-center gap-2 ${
                    selectedMode === "question"
                      ? "text-white/80"
                      : "text-zinc-500 dark:text-zinc-500"
                  }`}
                >
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path
                      fillRule="evenodd"
                      d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                      clipRule="evenodd"
                    />
                  </svg>
                  <span>Scene Comprehension</span>
                </div>
                <div
                  className={`flex items-center gap-2 ${
                    selectedMode === "question"
                      ? "text-white/80"
                      : "text-zinc-500 dark:text-zinc-500"
                  }`}
                >
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path
                      fillRule="evenodd"
                      d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                      clipRule="evenodd"
                    />
                  </svg>
                  <span>Multiple Choice</span>
                </div>
                <div
                  className={`flex items-center gap-2 ${
                    selectedMode === "question"
                      ? "text-white/80"
                      : "text-zinc-500 dark:text-zinc-500"
                  }`}
                >
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path
                      fillRule="evenodd"
                      d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                      clipRule="evenodd"
                    />
                  </svg>
                  <span>Image & Video Questions</span>
                </div>
              </div>
            </div>
          </button>

          {/* Human Evaluation Mode */}
          <button
            onClick={() => handleModeSelect("evaluation")}
            className={`p-8 rounded-2xl shadow-xl transition-all transform hover:scale-105 ${
              selectedMode === "evaluation"
                ? "bg-gradient-to-br from-emerald-500 to-teal-500 text-white ring-4 ring-emerald-300"
                : "bg-white dark:bg-zinc-800 hover:shadow-2xl"
            }`}
          >
            <div className="flex flex-col items-center">
              {/* Icon */}
              <div className="mb-4">
                <svg
                  className={`w-20 h-20 ${
                    selectedMode === "evaluation"
                      ? "text-white"
                      : "text-emerald-500"
                  }`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
              </div>

              {/* Title */}
              <h2
                className={`text-2xl font-bold mb-2 ${
                  selectedMode === "evaluation"
                    ? "text-white"
                    : "text-zinc-900 dark:text-zinc-50"
                }`}
              >
                Human Evaluation
              </h2>

              {/* Description */}
              <p
                className={`text-sm ${
                  selectedMode === "evaluation"
                    ? "text-white/90"
                    : "text-zinc-600 dark:text-zinc-400"
                }`}
              >
                Select the correct answer for benchmark evaluation
              </p>

              {/* Features */}
              <div className="mt-4 space-y-2 text-sm">
                <div
                  className={`flex items-center gap-2 ${
                    selectedMode === "evaluation"
                      ? "text-white/80"
                      : "text-zinc-500 dark:text-zinc-500"
                  }`}
                >
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path
                      fillRule="evenodd"
                      d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                      clipRule="evenodd"
                    />
                  </svg>
                  <span>Answer Selection</span>
                </div>
                <div
                  className={`flex items-center gap-2 ${
                    selectedMode === "evaluation"
                      ? "text-white/80"
                      : "text-zinc-500 dark:text-zinc-500"
                  }`}
                >
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path
                      fillRule="evenodd"
                      d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                      clipRule="evenodd"
                    />
                  </svg>
                  <span>Progress Tracking</span>
                </div>
                <div
                  className={`flex items-center gap-2 ${
                    selectedMode === "evaluation"
                      ? "text-white/80"
                      : "text-zinc-500 dark:text-zinc-500"
                  }`}
                >
                  <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path
                      fillRule="evenodd"
                      d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                      clipRule="evenodd"
                    />
                  </svg>
                  <span>Benchmark Dataset</span>
                </div>
              </div>
            </div>
          </button>
        </div>

        {/* Back to Login (Optional) */}
        <div className="text-center mt-8">
          <button
            onClick={handleLogout}
            className="text-sm text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-zinc-200 transition-colors"
          >
            ← Logout
          </button>
        </div>
      </div>
    </div>
  );
}
