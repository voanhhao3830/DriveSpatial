"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";

// ── Types ────────────────────────────────────────────────────────────────────

interface EvaluationQuestion {
  id: number;
  question_id: string;
  question_category: string;
  question_type: string;
  question_text: string;
  options: string[] | null;
  correct_answer: number;  // Float for absolute types, index for multiple choice
  source: string | null;
  scene_id: string | null;
  media_type: string;
  images: string[] | null;
  videos: string[] | null;
  frame_paths: string[] | null;
  metadata: Record<string, unknown> | null;
  status: string;
  time_spent_review: number;
  human_answer: number | null;  // Float for absolute types, index for multiple choice
  user_time_spent: number | null;
  created_at: string;
}

// Question types that require float input instead of multiple choice
const FLOAT_ANSWER_TYPES = ["distance_absolute", "scene_counting_absolute"];

interface EvaluationStats {
  total: number;
  answered: number;
  unanswered: number;
  total_time_spent: number;
  avg_time_per_question: number;
  username: string | null;
}

// ── Constants ────────────────────────────────────────────────────────────────

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const HPC_PREFIX = "";

const CATEGORY_COLORS: Record<string, { active: string; inactive: string }> = {
  causality: {
    active: "bg-orange-100 text-orange-700 border-orange-300",
    inactive: "bg-orange-50 text-orange-600 border-orange-200 hover:bg-orange-100",
  },
  construct: {
    active: "bg-blue-100 text-blue-700 border-blue-300",
    inactive: "bg-blue-50 text-blue-600 border-blue-200 hover:bg-blue-100",
  },
  relation: {
    active: "bg-green-100 text-green-700 border-green-300",
    inactive: "bg-green-50 text-green-600 border-green-200 hover:bg-green-100",
  },
};

const QUESTION_TYPES_BY_CATEGORY: Record<string, string[]> = {
  causality: [
    "action_reasoning",
    "event_ordering",
    "object_disappearance",
    "object_interaction",
    "object_manipulation",
    "occlusion_awareness",
  ],
  construct: [
    "cognitive_scene",
    "perspective_camera_matching",
    "ego_rotation",
    "camera_ordering",
    "camera_memory",
  ],
  relation: [
    "direction",
    "distance",
    "multi_steps_reasoning",
    "spatial_compability",
    "allocentric_imagination",
    "multiple_object_matching",
    "multiple_view_depth_awareness",
    "distance_absolute",
    "scene_counting_absolute",
  ],
};

// ── Helper: HPC image URL ────────────────────────────────────────────────────

function hpcImageUrl(relativePath: string): string {
  const fullPath = `${HPC_PREFIX}/${relativePath}`;
  return `/api/hpc-image?path=${encodeURIComponent(fullPath)}`;
}

// ── Main Component ───────────────────────────────────────────────────────────

export default function EvaluationPage() {
  const router = useRouter();

  // Auth
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [username, setUsername] = useState<string>("");

  // Data
  const [currentQuestion, setCurrentQuestion] = useState<EvaluationQuestion | null>(null);
  const [totalQuestions, setTotalQuestions] = useState(0);
  const [totalUnfiltered, setTotalUnfiltered] = useState(0);
  const [currentOffset, setCurrentOffset] = useState(0);

  // UI state
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);  // For multiple choice (index)
  const [floatAnswer, setFloatAnswer] = useState<string>("");  // For float input (as string for controlled input)
  const [showCorrectAnswer, setShowCorrectAnswer] = useState(false);

  // Filters
  const [filterCategory, setFilterCategory] = useState<string | null>(null);
  const [filterType, setFilterType] = useState<string | null>(null);
  const [filterSource, setFilterSource] = useState<string | null>(null);
  const [filterAnswered, setFilterAnswered] = useState<boolean | null>(false); // default: unanswered
  const [availableSources, setAvailableSources] = useState<{ source: string; count: number }[]>([]);

  // Timer
  const timerStartRef = useRef<number>(Date.now());
  const [timerDisplay, setTimerDisplay] = useState("00:00");

  // Stats
  const [stats, setStats] = useState<EvaluationStats | null>(null);

  // ── Auth check ─────────────────────────────────────────────────────────────

  useEffect(() => {
    const storedAuth = localStorage.getItem("drivespatial_auth");
    if (storedAuth) {
      try {
        const { username: u, timestamp } = JSON.parse(storedAuth);
        const sevenDaysInMs = 7 * 24 * 60 * 60 * 1000;
        if (Date.now() - timestamp < sevenDaysInMs && u) {
          setIsAuthenticated(true);
          setUsername(u);
        } else {
          localStorage.removeItem("drivespatial_auth");
          router.push("/");
        }
      } catch {
        localStorage.removeItem("drivespatial_auth");
        router.push("/");
      }
    } else {
      router.push("/");
    }
  }, [router]);

  // ── Fetch sources and stats ─────────────────────────────────────────────────

  const fetchStats = useCallback(() => {
    if (!username) return;
    fetch(`${API_BASE}/api/v1/evaluation/stats?username=${encodeURIComponent(username)}`)
      .then((r) => r.json())
      .then((data) => {
        setStats(data);
        setTotalUnfiltered(data.total || 0);
      })
      .catch(console.error);
  }, [username]);

  useEffect(() => {
    if (!isAuthenticated || !username) return;

    // Fetch sources
    fetch(`${API_BASE}/api/v1/evaluation/sources/list`)
      .then((r) => r.json())
      .then((data) => setAvailableSources(data.sources || []))
      .catch(console.error);

    // Fetch stats for this user
    fetchStats();
  }, [isAuthenticated, username, fetchStats]);

  // ── Load question ──────────────────────────────────────────────────────────

  const loadQuestion = useCallback(
    async (offset: number, category?: string | null, type?: string | null, source?: string | null, answered?: boolean | null, user?: string | null) => {
      setLoading(true);
      setError(null);

      const params = new URLSearchParams();
      params.set("top_k", "1");
      params.set("offset", String(offset));
      if (category) params.set("question_category", category);
      if (type) params.set("question_type", type);
      if (source) params.set("source", source);
      if (answered !== null) params.set("answered", String(answered));
      if (user) params.set("username", user);

      try {
        const res = await fetch(`${API_BASE}/api/v1/evaluation?${params.toString()}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();

        setTotalQuestions(json.total);
        if (json.data && json.data.length > 0) {
          const question = json.data[0];
          setCurrentQuestion(question);
          // Handle different answer types based on question type
          if (FLOAT_ANSWER_TYPES.includes(question.question_type)) {
            setFloatAnswer(question.human_answer !== null ? String(question.human_answer) : "");
            setSelectedAnswer(null);
          } else {
            setSelectedAnswer(question.human_answer);
            setFloatAnswer("");
          }
          setShowCorrectAnswer(false); // Reset when loading new question
        } else {
          setCurrentQuestion(null);
          setError("No questions found matching filters.");
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load question");
        setCurrentQuestion(null);
      } finally {
        setLoading(false);
      }
    },
    []
  );

  // ── Reload on filter change ────────────────────────────────────────────────

  useEffect(() => {
    if (!isAuthenticated || !username) return;
    setCurrentOffset(0);
    loadQuestion(0, filterCategory, filterType, filterSource, filterAnswered, username);
  }, [isAuthenticated, username, filterCategory, filterType, filterSource, filterAnswered, loadQuestion]);

  // ── Reset timer on question change ─────────────────────────────────────────

  useEffect(() => {
    timerStartRef.current = Date.now();
    setTimerDisplay("00:00");
  }, [currentQuestion?.question_id]);

  // ── Timer tick ─────────────────────────────────────────────────────────────

  useEffect(() => {
    const interval = setInterval(() => {
      const elapsed = Math.floor((Date.now() - timerStartRef.current) / 1000);
      const mins = String(Math.floor(elapsed / 60)).padStart(2, "0");
      const secs = String(elapsed % 60).padStart(2, "0");
      setTimerDisplay(`${mins}:${secs}`);
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  // ── Submit answer ──────────────────────────────────────────────────────────

  const handleSubmitAnswer = async () => {
    if (!currentQuestion || !username) return;
    
    // Determine the answer value based on question type
    const isFloatType = FLOAT_ANSWER_TYPES.includes(currentQuestion.question_type);
    let answerValue: number;
    
    if (isFloatType) {
      const parsed = parseFloat(floatAnswer);
      if (isNaN(parsed)) return;  // Invalid float input
      answerValue = parsed;
    } else {
      if (selectedAnswer === null) return;
      answerValue = selectedAnswer;
    }

    setSubmitting(true);
    const timeSpent = (Date.now() - timerStartRef.current) / 1000;

    try {
      const res = await fetch(`${API_BASE}/api/v1/evaluation/${currentQuestion.question_id}/answer`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          human_answer: answerValue,
          username: username,
          time_spent: timeSpent,
        }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      // Refresh stats for this user
      fetchStats();

      // If filtering unanswered, reload same offset (next unanswered)
      if (filterAnswered === false) {
        loadQuestion(currentOffset, filterCategory, filterType, filterSource, filterAnswered, username);
      } else {
        // Move to next question
        if (currentOffset < totalQuestions - 1) {
          const nextOffset = currentOffset + 1;
          setCurrentOffset(nextOffset);
          loadQuestion(nextOffset, filterCategory, filterType, filterSource, filterAnswered, username);
        } else {
          // Update in place
          setCurrentQuestion((prev) => (prev ? { ...prev, human_answer: answerValue, user_time_spent: timeSpent } : null));
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to submit answer");
    } finally {
      setSubmitting(false);
    }
  };

  // ── Delete question (admin only) ─────────────────────────────────────────────

  const handleDeleteQuestion = async () => {
    if (!currentQuestion || username !== "admin") return;

    const confirmed = window.confirm(
      `Are you sure you want to delete this question?\n\nQuestion ID: ${currentQuestion.question_id}\nType: ${currentQuestion.question_type}\n\nThis action cannot be undone.`
    );

    if (!confirmed) return;

    setDeleting(true);
    try {
      const res = await fetch(`${API_BASE}/api/v1/evaluation/${currentQuestion.question_id}`, {
        method: "DELETE",
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }

      // Refresh stats
      fetchStats();

      // Reload current offset (will show next question)
      if (totalQuestions > 1) {
        // If we're at the last question, go back one
        const newOffset = currentOffset >= totalQuestions - 1 ? Math.max(0, currentOffset - 1) : currentOffset;
        setCurrentOffset(newOffset);
        loadQuestion(newOffset, filterCategory, filterType, filterSource, filterAnswered, username);
      } else {
        // No more questions
        setCurrentQuestion(null);
        setTotalQuestions(0);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to delete question");
    } finally {
      setDeleting(false);
    }
  };

  // ── Navigation ─────────────────────────────────────────────────────────────

  const goNext = () => {
    if (currentOffset < totalQuestions - 1) {
      const next = currentOffset + 1;
      setCurrentOffset(next);
      loadQuestion(next, filterCategory, filterType, filterSource, filterAnswered, username);
    }
  };

  const goPrev = () => {
    if (currentOffset > 0) {
      const prev = currentOffset - 1;
      setCurrentOffset(prev);
      loadQuestion(prev, filterCategory, filterType, filterSource, filterAnswered, username);
    }
  };

  // ── Helper: format time ────────────────────────────────────────────────────

  const formatTime = (seconds: number): string => {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const mins = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    if (mins < 60) return `${mins}m ${secs}s`;
    const hours = Math.floor(mins / 60);
    const remainMins = mins % 60;
    return `${hours}h ${remainMins}m`;
  };

  // ── Computed ───────────────────────────────────────────────────────────────

  const isFiltered = filterCategory || filterType || filterSource || filterAnswered !== null;
  const progressPct = totalQuestions > 0 ? ((currentOffset + 1) / totalQuestions) * 100 : 0;

  // Sidebar collapse state
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // ── Render ─────────────────────────────────────────────────────────────────

  if (!isAuthenticated) {
    return (
      <div className="h-screen w-screen bg-zinc-950 flex items-center justify-center">
        <div className="animate-spin rounded-full h-10 w-10 border-4 border-zinc-700 border-t-emerald-500" />
      </div>
    );
  }

  return (
    <div className="h-screen w-screen bg-zinc-50 dark:bg-zinc-950 font-sans flex overflow-hidden">
      {/* ══ SIDEBAR ══════════════════════════════════════════════════════════ */}
      <aside className={`${sidebarCollapsed ? "w-12" : "w-64"} flex-shrink-0 bg-white dark:bg-zinc-900 border-r border-zinc-200 dark:border-zinc-800 flex flex-col overflow-hidden transition-all duration-300`}>
        {/* Collapse toggle */}
        <button
          onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
          className="absolute left-0 top-1/2 -translate-y-1/2 z-10 w-5 h-10 bg-zinc-200 dark:bg-zinc-700 hover:bg-zinc-300 dark:hover:bg-zinc-600 rounded-r-md flex items-center justify-center transition-all"
          style={{ marginLeft: sidebarCollapsed ? "48px" : "256px" }}
        >
          <svg
            className={`w-3 h-3 text-zinc-600 dark:text-zinc-300 transition-transform ${sidebarCollapsed ? "" : "rotate-180"}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </button>

        {/* Header */}
        <div className={`p-4 border-b border-zinc-200 dark:border-zinc-800 ${sidebarCollapsed ? "hidden" : ""}`}>
          <div className="flex items-center gap-2 mb-1">
            <svg className="w-5 h-5 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <h1 className="text-lg font-bold text-zinc-900 dark:text-zinc-50">Human Evaluation</h1>
          </div>
          <p className="text-xs text-zinc-500">Select the correct answer</p>
        </div>
        {/* Collapsed header icon */}
        {sidebarCollapsed && (
          <div className="p-3 border-b border-zinc-200 dark:border-zinc-800 flex justify-center">
            <svg className="w-5 h-5 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
        )}

        {/* Stats */}
        {stats && !sidebarCollapsed && (
          <div className="p-4 border-b border-zinc-200 dark:border-zinc-800">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Your Progress</h3>
              <span className="text-[10px] text-zinc-500 bg-zinc-100 dark:bg-zinc-800 px-1.5 py-0.5 rounded">{username}</span>
            </div>
            {/* <div className="grid grid-cols-3 gap-2 text-center">
              <div className="bg-zinc-100 dark:bg-zinc-800 rounded-lg p-2">
                <div className="text-lg font-bold text-emerald-600 dark:text-emerald-400">{stats.answered}</div>
                <div className="text-[10px] text-zinc-500">Done</div>
              </div>
              <div className="bg-zinc-100 dark:bg-zinc-800 rounded-lg p-2">
                <div className="text-lg font-bold text-amber-600 dark:text-amber-400">{stats.unanswered}</div>
                <div className="text-[10px] text-zinc-500">Left</div>
              </div>
              <div className="bg-zinc-100 dark:bg-zinc-800 rounded-lg p-2">
                <div className="text-lg font-bold text-zinc-900 dark:text-zinc-50">{stats.total}</div>
                <div className="text-[10px] text-zinc-500">Total</div>
              </div>
            </div> */}
            <div className="mt-2 h-2 bg-zinc-200 dark:bg-zinc-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-emerald-500 transition-all duration-300"
                style={{ width: `${stats.total > 0 ? (stats.answered / stats.total) * 100 : 0}%` }}
              />
            </div>
            {/* Time stats */}
            {/* <div className="mt-3 grid grid-cols-2 gap-2 text-center">
              <div className="bg-zinc-50 dark:bg-zinc-800/50 rounded-lg p-2 border border-zinc-200 dark:border-zinc-700">
                <div className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">
                  {formatTime(stats.total_time_spent)}
                </div>
                <div className="text-[10px] text-zinc-500">Total Time</div>
              </div>
              <div className="bg-zinc-50 dark:bg-zinc-800/50 rounded-lg p-2 border border-zinc-200 dark:border-zinc-700">
                <div className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">
                  {formatTime(stats.avg_time_per_question)}
                </div>
                <div className="text-[10px] text-zinc-500">Avg / Question</div>
              </div>
            </div> */}
          </div>
        )}

        {/* Filters */}
        <div className={`flex-1 overflow-y-auto p-4 space-y-4 ${sidebarCollapsed ? "hidden" : ""}`}>
          {/* Category filter */}
          <div>
            <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-2">Category</h3>
            <div className="flex flex-wrap gap-1.5">
              {["causality", "construct", "relation"].map((cat) => (
                <button
                  key={cat}
                  onClick={() => {
                    setFilterCategory(filterCategory === cat ? null : cat);
                    setFilterType(null);
                  }}
                  className={`px-2.5 py-1 rounded-full text-xs font-medium border transition-colors ${
                    filterCategory === cat
                      ? CATEGORY_COLORS[cat]?.active
                      : CATEGORY_COLORS[cat]?.inactive
                  }`}
                >
                  {cat}
                </button>
              ))}
            </div>
          </div>

          {/* Type filter */}
          <div>
            <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-2">Question Type</h3>
            <select
              value={filterType || ""}
              onChange={(e) => setFilterType(e.target.value || null)}
              className="w-full px-3 py-2 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-800 text-sm text-zinc-900 dark:text-zinc-100"
            >
              <option value="">All types</option>
              {(filterCategory
                ? QUESTION_TYPES_BY_CATEGORY[filterCategory] || []
                : Object.values(QUESTION_TYPES_BY_CATEGORY).flat()
              ).map((t) => (
                <option key={t} value={t}>
                  {t.replace(/_/g, " ")}
                </option>
              ))}
            </select>
          </div>

          {/* Source filter */}
          <div>
            <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-2">Source</h3>
            <select
              value={filterSource || ""}
              onChange={(e) => setFilterSource(e.target.value || null)}
              className="w-full px-3 py-2 rounded-lg border border-zinc-200 dark:border-zinc-700 bg-white dark:bg-zinc-800 text-sm text-zinc-900 dark:text-zinc-100"
            >
              <option value="">All sources</option>
              {availableSources.map((s) => (
                <option key={s.source} value={s.source}>
                  {s.source} ({s.count})
                </option>
              ))}
            </select>
          </div>

          {/* Answered filter */}
          <div>
            <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-2">Status</h3>
            <div className="flex flex-col gap-1">
              {[
                { value: false, label: "Unanswered", color: "text-amber-600" },
                { value: true, label: "Answered", color: "text-emerald-600" },
                { value: null, label: "All", color: "text-zinc-600" },
              ].map((opt) => (
                <button
                  key={String(opt.value)}
                  onClick={() => setFilterAnswered(opt.value)}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm transition-colors ${
                    filterAnswered === opt.value
                      ? "bg-zinc-100 dark:bg-zinc-800 font-medium"
                      : "hover:bg-zinc-50 dark:hover:bg-zinc-800/50"
                  }`}
                >
                  <span
                    className={`w-2 h-2 rounded-full ${
                      filterAnswered === opt.value ? "bg-emerald-500" : "bg-zinc-300 dark:bg-zinc-600"
                    }`}
                  />
                  <span className={opt.color}>{opt.label}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className={`p-4 border-t border-zinc-200 dark:border-zinc-800 ${sidebarCollapsed ? "hidden" : ""}`}>
          <div className="text-xs text-zinc-500 mb-2">
            {currentOffset + 1} / {totalQuestions} questions
            {isFiltered && totalUnfiltered > 0 && (
              <span className="text-zinc-400"> of {totalUnfiltered} total</span>
            )}
          </div>
          <button
            onClick={() => router.push("/mode-select")}
            className="w-full px-3 py-2 rounded-lg border border-zinc-200 dark:border-zinc-700 text-sm text-zinc-600 dark:text-zinc-400 hover:bg-zinc-50 dark:hover:bg-zinc-800 transition-colors"
          >
            ← Back to Mode Select
          </button>
        </div>
        {/* Collapsed footer */}
        {sidebarCollapsed && (
          <div className="p-2 border-t border-zinc-200 dark:border-zinc-800 flex justify-center">
            <button
              onClick={() => router.push("/mode-select")}
              className="p-2 rounded-lg border border-zinc-200 dark:border-zinc-700 text-zinc-600 dark:text-zinc-400 hover:bg-zinc-50 dark:hover:bg-zinc-800 transition-colors"
              title="Back to Mode Select"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
            </button>
          </div>
        )}
      </aside>

      {/* ══ MAIN CONTENT ═════════════════════════════════════════════════════ */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="bg-white dark:bg-zinc-900 border-b border-zinc-200 dark:border-zinc-800 flex-shrink-0">
          <div className="px-8 py-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              {currentQuestion && (
                <>
                  <span
                    className={`px-2.5 py-0.5 rounded-full text-xs font-semibold border ${
                      CATEGORY_COLORS[currentQuestion.question_category]?.active ?? "bg-zinc-200 text-zinc-700 border-zinc-300"
                    }`}
                  >
                    {currentQuestion.question_category}
                  </span>
                  <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 border border-zinc-200 dark:border-zinc-700">
                    {currentQuestion.question_type.replace(/_/g, " ")}
                  </span>
                  {currentQuestion.scene_id && (
                    <span className="px-2.5 py-0.5 rounded-full text-xs font-mono bg-violet-100 dark:bg-violet-900/30 text-violet-700 dark:text-violet-400 border border-violet-200 dark:border-violet-700">
                      Scene: {currentQuestion.scene_id}
                    </span>
                  )}
                </>
              )}
            </div>

            <div className="flex items-center gap-4">
              <span className="text-sm font-mono text-zinc-500">{timerDisplay}</span>
              <span className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">
                {currentOffset + 1}
                <span className="font-normal text-zinc-400"> / {totalQuestions}</span>
              </span>
            </div>
          </div>

          {/* Progress bar */}
          <div className="h-1 w-full bg-zinc-100 dark:bg-zinc-800">
            <div
              className={`h-full transition-all duration-300 ${loading ? "animate-pulse bg-zinc-300 dark:bg-zinc-600 w-full" : "bg-emerald-500"}`}
              style={loading ? undefined : { width: `${progressPct}%` }}
            />
          </div>
        </header>

        {/* Body */}
        <div className="flex-1 flex overflow-hidden min-h-0">
          {loading && (
            <div className="flex-1 flex items-center justify-center">
              <div className="animate-spin rounded-full h-10 w-10 border-4 border-zinc-200 border-t-emerald-500" />
            </div>
          )}

          {!loading && error && (
            <div className="flex-1 flex items-center justify-center">
              <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-700 text-red-700 dark:text-red-400 px-6 py-4 rounded-lg text-sm max-w-md text-center">
                {error}
              </div>
            </div>
          )}

          {!loading && !error && currentQuestion && (
            <>
              {/* Left: Images */}
              <div className="flex-1 overflow-y-auto bg-zinc-950 border-r border-zinc-800 p-5 space-y-4 min-w-0">
                <ImagePanel
                  question={currentQuestion}
                  selectedAnswer={selectedAnswer}
                  onSelectAnswer={setSelectedAnswer}
                />
              </div>

              {/* Right: Question + Options */}
              <div className="w-[480px] flex-shrink-0 overflow-y-auto p-6 space-y-4 bg-zinc-50 dark:bg-zinc-950">
                {/* Answered badge */}
                {currentQuestion.human_answer !== null && (
                  <div className="flex items-center justify-between px-4 py-2 rounded-xl bg-emerald-50 dark:bg-emerald-900/20 border border-emerald-300 dark:border-emerald-700 text-emerald-700 dark:text-emerald-400 text-sm">
                    <div className="flex items-center gap-2">
                      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      <span>
                        Your answer: <strong>
                          {FLOAT_ANSWER_TYPES.includes(currentQuestion.question_type)
                            ? currentQuestion.human_answer
                            : `Option ${String.fromCharCode(65 + currentQuestion.human_answer)}`}
                        </strong>
                      </span>
                    </div>
                    {currentQuestion.user_time_spent !== null && currentQuestion.user_time_spent > 0 && (
                      <span className="text-xs text-emerald-600 dark:text-emerald-500 bg-emerald-100 dark:bg-emerald-900/40 px-2 py-0.5 rounded">
                        {formatTime(currentQuestion.user_time_spent)}
                      </span>
                    )}
                  </div>
                )}

                {/* Question text */}
                <div className="bg-white dark:bg-zinc-900 rounded-xl shadow-sm border border-zinc-200 dark:border-zinc-800 p-5">
                  <QuestionText text={currentQuestion.question_text} />
                </div>

                {/* Options */}
                {(() => {
                  // Question types that need image-based options
                  const IMAGE_OPTION_TYPES = ["cognitive_scene", "perspective_camera_matching"];
                  const needsImageOptions = IMAGE_OPTION_TYPES.includes(currentQuestion.question_type);
                  const needsFloatInput = FLOAT_ANSWER_TYPES.includes(currentQuestion.question_type);
                  const hasTextOptions = currentQuestion.options && currentQuestion.options.length > 0;

                  if (needsFloatInput) {
                    // Show float input for distance_absolute and scene_counting_absolute
                    const isDistanceType = currentQuestion.question_type === "distance_absolute";
                    const placeholder = isDistanceType ? "Enter distance (e.g., 15.5)" : "Enter count (e.g., 3)";
                    const label = isDistanceType ? "Distance (meters)" : "Count";
                    
                    return (
                      <div className="bg-white dark:bg-zinc-900 rounded-xl shadow-sm border border-zinc-200 dark:border-zinc-800 p-5">
                        <div className="flex items-center justify-between mb-3">
                          <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
                            Enter your answer
                          </h3>
                          {showCorrectAnswer && (
                            <span className="text-xs font-medium text-amber-600 dark:text-amber-400 bg-amber-100 dark:bg-amber-900/30 px-2 py-0.5 rounded">
                              Correct: {currentQuestion.correct_answer}
                            </span>
                          )}
                        </div>
                        <div className="space-y-3">
                          <label className="block">
                            <span className="text-sm font-medium text-zinc-700 dark:text-zinc-300">{label}</span>
                            <input
                              type="number"
                              step="any"
                              value={floatAnswer}
                              onChange={(e) => setFloatAnswer(e.target.value)}
                              placeholder={placeholder}
                              className={`mt-1 block w-full px-4 py-3 rounded-lg border-2 text-lg font-mono transition-all focus:outline-none focus:ring-2 focus:ring-emerald-500 ${
                                floatAnswer && !isNaN(parseFloat(floatAnswer))
                                  ? "border-emerald-500 bg-emerald-50 dark:bg-emerald-900/20"
                                  : "border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800"
                              } text-zinc-900 dark:text-zinc-100`}
                            />
                          </label>
                          {floatAnswer && isNaN(parseFloat(floatAnswer)) && (
                            <p className="text-sm text-red-500">Please enter a valid number</p>
                          )}
                          {floatAnswer && !isNaN(parseFloat(floatAnswer)) && (
                            <div className="flex items-center gap-2 text-sm text-emerald-600 dark:text-emerald-400">
                              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                              </svg>
                              <span>Your answer: <strong>{parseFloat(floatAnswer)}</strong></span>
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  } else if (needsImageOptions) {
                    // Show image options panel for these question types
                    return (
                      <ImageOptionsPanel
                        question={currentQuestion}
                        selectedAnswer={selectedAnswer}
                        onSelectAnswer={setSelectedAnswer}
                        showCorrectAnswer={showCorrectAnswer}
                      />
                    );
                  } else if (hasTextOptions) {
                    // Show text options for other question types
                    return (
                      <div className="bg-white dark:bg-zinc-900 rounded-xl shadow-sm border border-zinc-200 dark:border-zinc-800 p-5">
                        <div className="flex items-center justify-between mb-3">
                          <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
                            Select your answer
                          </h3>
                          {showCorrectAnswer && (
                            <span className="text-xs font-medium text-amber-600 dark:text-amber-400 bg-amber-100 dark:bg-amber-900/30 px-2 py-0.5 rounded">
                              Correct: {String.fromCharCode(65 + currentQuestion.correct_answer)}
                            </span>
                          )}
                        </div>
                        <div className="space-y-2">
                          {currentQuestion.options!.map((option, idx) => {
                            const isSelected = selectedAnswer === idx;
                            const isCorrect = showCorrectAnswer && idx === currentQuestion.correct_answer;
                            return (
                              <button
                                key={idx}
                                onClick={() => setSelectedAnswer(idx)}
                                className={`w-full flex items-start gap-3 p-3 rounded-lg border-2 text-left transition-all ${
                                  isCorrect
                                    ? "border-amber-500 bg-amber-50 dark:bg-amber-900/20"
                                    : isSelected
                                    ? "border-emerald-500 bg-emerald-50 dark:bg-emerald-900/20"
                                    : "border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800/50 hover:border-zinc-300 dark:hover:border-zinc-600"
                                }`}
                              >
                                <div
                                  className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                                    isCorrect
                                      ? "bg-amber-500 text-white"
                                      : isSelected
                                      ? "bg-emerald-500 text-white"
                                      : "bg-zinc-200 dark:bg-zinc-700 text-zinc-600 dark:text-zinc-300"
                                  }`}
                                >
                                  {isCorrect ? "✓" : String.fromCharCode(65 + idx)}
                                </div>
                                <span className={`text-sm flex-1 ${
                                  isCorrect
                                    ? "text-amber-700 dark:text-amber-300 font-medium"
                                    : isSelected
                                    ? "text-emerald-700 dark:text-emerald-300 font-medium"
                                    : "text-zinc-700 dark:text-zinc-300"
                                }`}>
                                  {option}
                                </span>
                                {isCorrect && (
                                  <span className="text-xs font-medium text-amber-600 dark:text-amber-400">
                                    Correct
                                  </span>
                                )}
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    );
                  } else {
                    // No options available
                    return (
                      <div className="bg-white dark:bg-zinc-900 rounded-xl shadow-sm border border-zinc-200 dark:border-zinc-800 p-5">
                        <p className="text-sm text-zinc-500">No options available for this question.</p>
                      </div>
                    );
                  }
                })()}
              </div>
            </>
          )}
        </div>

        {/* Footer */}
        <footer className="bg-white dark:bg-zinc-900 border-t border-zinc-200 dark:border-zinc-800 px-8 py-4 flex items-center justify-between flex-shrink-0">
          <div className="flex items-center gap-3">
            <button
              onClick={goPrev}
              disabled={currentOffset === 0 || loading}
              className="px-4 py-2 rounded-lg border border-zinc-200 dark:border-zinc-700 text-sm font-medium text-zinc-600 dark:text-zinc-400 hover:bg-zinc-50 dark:hover:bg-zinc-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              ← Prev
            </button>
            <button
              onClick={goNext}
              disabled={currentOffset >= totalQuestions - 1 || loading}
              className="px-4 py-2 rounded-lg border border-zinc-200 dark:border-zinc-700 text-sm font-medium text-zinc-600 dark:text-zinc-400 hover:bg-zinc-50 dark:hover:bg-zinc-800 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
            >
              Next →
            </button>
          </div>

          <div className="flex items-center gap-3">
            {/* Show Answer button - only visible for admin user "admin" */}
            {username === "admin" && currentQuestion && (
              <button
                onClick={() => setShowCorrectAnswer(!showCorrectAnswer)}
                className={`px-4 py-2.5 rounded-lg font-semibold text-sm transition-colors flex items-center gap-2 ${
                  showCorrectAnswer
                    ? "bg-amber-500 hover:bg-amber-600 text-white"
                    : "bg-amber-100 hover:bg-amber-200 text-amber-700 border border-amber-300"
                }`}
                title="Show/Hide correct answer (Admin only)"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  {showCorrectAnswer ? (
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                  ) : (
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                  )}
                </svg>
                {showCorrectAnswer ? "Hide Answer" : "Show Answer"}
              </button>
            )}

            {username === "admin" && currentQuestion && (
              <button
                onClick={handleDeleteQuestion}
                disabled={deleting || loading}
                className="px-4 py-2.5 rounded-lg bg-red-500 hover:bg-red-600 text-white font-semibold text-sm disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
                title="Delete this question (Admin only)"
              >
                {deleting ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" />
                    Deleting...
                  </>
                ) : (
                  <>
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                    Delete
                  </>
                )}
              </button>
            )}

            <button
              onClick={handleSubmitAnswer}
              disabled={
                submitting || 
                loading || 
                (currentQuestion && FLOAT_ANSWER_TYPES.includes(currentQuestion.question_type) 
                  ? floatAnswer === "" || isNaN(parseFloat(floatAnswer))
                  : selectedAnswer === null)
              }
              className="px-6 py-2.5 rounded-lg bg-emerald-500 hover:bg-emerald-600 text-white font-semibold text-sm disabled:opacity-40 disabled:cursor-not-allowed transition-colors flex items-center gap-2"
            >
              {submitting ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" />
                  Submitting...
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  Submit Answer
                </>
              )}
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
}

// ── Sub-components ───────────────────────────────────────────────────────────

// Image lightbox modal for viewing images larger
function ImageLightbox({
  imagePath,
  alt,
  onClose,
}: {
  imagePath: string;
  alt: string;
  onClose: () => void;
}) {
  const fullUrl = `/api/hpc-image?path=${encodeURIComponent(`${HPC_PREFIX}/${imagePath}`)}`;

  useEffect(() => {
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handleEsc);
    return () => window.removeEventListener("keydown", handleEsc);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-sm"
      onClick={onClose}
    >
      <div className="relative max-w-[95vw] max-h-[95vh]" onClick={(e) => e.stopPropagation()}>
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute -top-10 right-0 text-white/70 hover:text-white transition-colors flex items-center gap-2 text-sm"
        >
          <span>Press ESC or click outside to close</span>
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
        {/* Image */}
        <img
          src={fullUrl}
          alt={alt}
          className="max-w-[95vw] max-h-[90vh] object-contain rounded-lg shadow-2xl"
        />
        {/* Label */}
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4 rounded-b-lg">
          <p className="text-white text-center font-medium">{alt}</p>
        </div>
      </div>
    </div>
  );
}

function ImagePanel({
  question,
  selectedAnswer,
  onSelectAnswer,
}: {
  question: EvaluationQuestion;
  selectedAnswer: number | null;
  onSelectAnswer: (idx: number) => void;
}) {
  const [lightboxImage, setLightboxImage] = useState<{ path: string; alt: string } | null>(null);

  const isVideo = question.media_type === "video";
  const hasImages = !isVideo && question.images && question.images.length > 0;
  const hasFrames = isVideo && question.frame_paths && question.frame_paths.length > 0;

  const handleContextMenu = useCallback((e: React.MouseEvent, path: string, alt: string) => {
    e.preventDefault();
    setLightboxImage({ path, alt });
  }, []);

  if (!hasImages && !hasFrames) {
    return (
      <div className="flex items-center justify-center h-full text-zinc-500 text-sm">
        No media available
      </div>
    );
  }

  const frameBase = isVideo
    ? `data/questions-v8/${question.question_category}/${question.question_type}/images/${question.question_id}`
    : null;

  // Helper to extract record_id from filename like "frame_0000_record_37895.jpg"
  const extractRecordId = (filename: string): number => {
    const match = filename.match(/record_(\d+)/);
    return match ? parseInt(match[1], 10) : 0;
  };

  let displayItems: { path: string; label: string; recordId?: number }[] = isVideo
    ? (question.frame_paths ?? []).map((fn) => ({
        path: `${frameBase}/${fn}`,
        label: fn,
        recordId: extractRecordId(fn),
      }))
    : (question.images ?? []).map((p) => ({
        path: p,
        label: p.split("/").pop() ?? p,
      }));

  // Sort by record_id for causality questions (video frames)
  if (question.question_category === "causality" && isVideo) {
    displayItems = [...displayItems].sort((a, b) => (a.recordId ?? 0) - (b.recordId ?? 0));
  }

  // Grid layout for cognitive_scene / perspective_camera_matching
  const GRID_TYPES = ["cognitive_scene", "perspective_camera_matching"];
  const isGridLayout = GRID_TYPES.includes(question.question_type) && displayItems.length >= 2;

  if (isGridLayout) {
    const [context] = displayItems;

    return (
      <>
        {/* Lightbox modal */}
        {lightboxImage && (
          <ImageLightbox
            imagePath={lightboxImage.path}
            alt={lightboxImage.alt}
            onClose={() => setLightboxImage(null)}
          />
        )}

        {/* Full-width context image only */}
        <div className="flex flex-col h-full">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs text-zinc-400 font-medium">Context Image</span>
            <span className="text-[10px] text-zinc-600 bg-zinc-800 px-2 py-0.5 rounded">
              Right-click to enlarge
            </span>
          </div>
          <div
            className="flex-1 min-h-0 cursor-zoom-in"
            onContextMenu={(e) => handleContextMenu(e, context.path, "Context Image")}
          >
            <HpcImage relativePath={context.path} alt="Context" fill />
          </div>
        </div>
      </>
    );
  }

  // Default: vertical list
  return (
    <>
      {/* Lightbox modal */}
      {lightboxImage && (
        <ImageLightbox
          imagePath={lightboxImage.path}
          alt={lightboxImage.alt}
          onClose={() => setLightboxImage(null)}
        />
      )}

      <div className="flex items-center gap-2 pb-1 border-b border-zinc-800">
        {isVideo ? (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-violet-900/40 border border-violet-700 text-violet-300 text-xs font-semibold">
            <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 20 20">
              <path d="M2 6a2 2 0 012-2h6a2 2 0 012 2v8a2 2 0 01-2 2H4a2 2 0 01-2-2V6zm12.553 1.106A1 1 0 0014 8v4a1 1 0 00.553.894l2 1A1 1 0 0018 13V7a1 1 0 00-1.447-.894l-2 1z" />
            </svg>
            Video — {displayItems.length} frames
          </span>
        ) : (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-zinc-800 border border-zinc-700 text-zinc-400 text-xs font-semibold">
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
            Image — {displayItems.length} {displayItems.length === 1 ? "file" : "files"}
          </span>
        )}
        <span className="text-[10px] text-zinc-600 ml-auto">Right-click to enlarge</span>
      </div>

      {displayItems.map((item, i) => (
        <div
          key={i}
          className="space-y-1.5 cursor-zoom-in"
          onContextMenu={(e) => handleContextMenu(e, item.path, `Image ${i + 1}`)}
        >
          <div className="flex items-center gap-1.5">
            <span className="w-4 h-4 rounded-full bg-zinc-700 flex items-center justify-center text-zinc-300 font-bold text-[9px]">
              {i + 1}
            </span>
            <span className="text-xs text-zinc-500 truncate">{item.label}</span>
            {item.recordId !== undefined && (
              <span className="text-[10px] text-violet-400 bg-violet-900/30 px-1.5 py-0.5 rounded ml-auto">
                #{item.recordId}
              </span>
            )}
          </div>
          <HpcImage relativePath={item.path} alt={`Frame ${i + 1} (Record #${item.recordId ?? ""})`} />
        </div>
      ))}
    </>
  );
}

// Image options panel for image-based questions (shown in right panel)
function ImageOptionsPanel({
  question,
  selectedAnswer,
  onSelectAnswer,
  showCorrectAnswer = false,
}: {
  question: EvaluationQuestion;
  selectedAnswer: number | null;
  onSelectAnswer: (idx: number) => void;
  showCorrectAnswer?: boolean;
}) {
  const [lightboxImage, setLightboxImage] = useState<{ path: string; alt: string } | null>(null);

  // Get option images (skip the first one which is context)
  const images = question.images ?? [];
  if (images.length < 2) {
    return (
      <div className="bg-white dark:bg-zinc-900 rounded-xl shadow-sm border border-zinc-200 dark:border-zinc-800 p-5">
        <p className="text-sm text-zinc-500">No option images available</p>
      </div>
    );
  }

  const [, ...optionImages] = images; // Skip context image

  // IMPORTANT: Do NOT sort the images!
  // The correct_answer index refers to the display order (Option A=0, B=1, C=2, D=3)
  // which matches the order in the images array, NOT the option number in filenames.
  // e.g., if images = [cameras.jpg, option_2.png, option_1.png, option_0.png, option_3.png]
  // then Option A = option_2.png, Option B = option_1.png, etc.
  // and correct_answer=2 means Option C = option_0.png

  // Use text options if available (e.g., "Camera A", "Camera B")
  const textOptions = question.options ?? [];

  const handleContextMenu = (e: React.MouseEvent, path: string, alt: string) => {
    e.preventDefault();
    setLightboxImage({ path, alt });
  };

  return (
    <>
      {/* Lightbox modal */}
      {lightboxImage && (
        <ImageLightbox
          imagePath={lightboxImage.path}
          alt={lightboxImage.alt}
          onClose={() => setLightboxImage(null)}
        />
      )}

      <div className="bg-white dark:bg-zinc-900 rounded-xl shadow-sm border border-zinc-200 dark:border-zinc-800 p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">
            Select your answer
          </h3>
          <div className="flex items-center gap-2">
            {showCorrectAnswer && (
              <span className="text-xs font-medium text-amber-600 dark:text-amber-400 bg-amber-100 dark:bg-amber-900/30 px-2 py-0.5 rounded">
                Correct: {String.fromCharCode(65 + question.correct_answer)}
              </span>
            )}
            <span className="text-[10px] text-zinc-500">Right-click to enlarge</span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3">
          {optionImages.slice(0, 4).map((imgPath, i) => {
            const isSelected = selectedAnswer === i;
            const isCorrect = showCorrectAnswer && i === question.correct_answer;
            // Use text option label if available, otherwise default to "Option A/B/C/D"
            const optionLabel = textOptions[i] || `Option ${String.fromCharCode(65 + i)}`;

            return (
              <button
                key={i}
                type="button"
                onClick={() => onSelectAnswer(i)}
                onContextMenu={(e) => handleContextMenu(e, imgPath, optionLabel)}
                className={`flex flex-col rounded-xl overflow-hidden transition-all ${
                  isCorrect
                    ? "ring-2 ring-amber-500 ring-offset-2 ring-offset-white dark:ring-offset-zinc-900"
                    : isSelected
                    ? "ring-2 ring-emerald-500 ring-offset-2 ring-offset-white dark:ring-offset-zinc-900"
                    : "ring-1 ring-zinc-200 dark:ring-zinc-700 hover:ring-2 hover:ring-zinc-400"
                }`}
              >
                {/* Option label */}
                <div
                  className={`flex items-center gap-2 px-3 py-2 ${
                    isCorrect
                      ? "bg-amber-500 text-white"
                      : isSelected
                      ? "bg-emerald-500 text-white"
                      : "bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300"
                  }`}
                >
                  <div
                    className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold ${
                      isCorrect
                        ? "bg-white text-amber-600"
                        : isSelected
                        ? "bg-white text-emerald-600"
                        : "bg-zinc-300 dark:bg-zinc-600 text-zinc-700 dark:text-zinc-200"
                    }`}
                  >
                    {isCorrect ? "✓" : String.fromCharCode(65 + i)}
                  </div>
                  <span className="text-xs font-medium">{optionLabel}</span>
                  {isCorrect && (
                    <span className="text-[10px] ml-auto font-semibold">Correct</span>
                  )}
                  {isSelected && !isCorrect && (
                    <svg className="w-4 h-4 ml-auto" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                  )}
                </div>
                {/* Option image */}
                <div className="aspect-square bg-zinc-50 dark:bg-zinc-800">
                  <HpcImage relativePath={imgPath} alt={optionLabel} fill />
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </>
  );
}

function HpcImage({
  relativePath,
  alt,
  fill = false,
}: {
  relativePath: string | null;
  alt: string;
  fill?: boolean;
}) {
  const [status, setStatus] = useState<"loading" | "loaded" | "error">("loading");

  if (!relativePath) {
    return (
      <div className={`flex items-center justify-center text-xs text-zinc-500 bg-zinc-800 rounded-xl ${fill ? "h-full" : "h-32"}`}>
        No image path
      </div>
    );
  }

  const src = hpcImageUrl(relativePath);

  return (
    <div className={`bg-zinc-800 rounded-xl overflow-hidden ${fill ? "h-full" : ""}`}>
      {status === "loading" && (
        <div className={`flex items-center justify-center ${fill ? "h-full" : "h-40"}`}>
          <div className="animate-spin rounded-full h-7 w-7 border-[3px] border-zinc-700 border-t-emerald-500" />
        </div>
      )}
      {status === "error" && (
        <div className={`flex flex-col items-center justify-center gap-1.5 text-xs text-zinc-500 ${fill ? "h-full" : "h-32"}`}>
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
          Failed to load
        </div>
      )}
      <img
        src={src}
        alt={alt}
        className={`w-full object-contain rounded-xl ${fill ? "h-full" : ""} ${status === "loaded" ? "block" : "hidden"}`}
        onLoad={() => setStatus("loaded")}
        onError={() => setStatus("error")}
      />
    </div>
  );
}

function QuestionText({ text }: { text: string }) {
  // Replace <image> and <video> tokens with badges
  const parts = text.split(/(<image>|<video>)/g);
  let imageIdx = 0;

  return (
    <div className="text-sm text-zinc-700 dark:text-zinc-300 leading-relaxed whitespace-pre-wrap">
      {parts.map((part, i) => {
        if (part === "<image>") {
          imageIdx++;
          return (
            <span
              key={i}
              className="inline-flex items-center justify-center w-5 h-5 rounded bg-blue-500 text-white text-[10px] font-bold mx-0.5"
            >
              {imageIdx}
            </span>
          );
        }
        if (part === "<video>") {
          return (
            <span
              key={i}
              className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-violet-500 text-white text-[10px] font-bold mx-0.5"
            >
              ▶ video
            </span>
          );
        }
        return <span key={i}>{part}</span>;
      })}
    </div>
  );
}
