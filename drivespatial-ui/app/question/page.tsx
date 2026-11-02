"use client";

import { useState, useEffect, useCallback, useRef, Fragment } from "react";
import { useRouter } from "next/navigation";

const HPC_PREFIX = "";

/** Build the Next.js proxy URL for an HPC image given its relative path. */
function hpcImageUrl(relativePath: string): string {
  const fullPath = `${HPC_PREFIX}/${relativePath}`;
  return `/api/hpc-image?path=${encodeURIComponent(fullPath)}`;
}

// ── Type definitions ───────────────────────────────────────────────────────────

type QuestionStatus = "pending" | "accepted" | "rejected" | "ambiguous";

interface Question {
  id: number;
  question_id: string;
  question_category: string;
  question_type: string;
  question_text: string;
  options: string[] | null;
  correct_answer: number;
  source: string | null;
  scene_id: string | null;
  media_type: string;
  images: string[] | null;
  videos: string[] | null;
  frame_paths: string[] | null;
  metadata: Record<string, any> | null;
  status: QuestionStatus;
  time_spent_review: number;
  created_at: string;
}

interface QuestionResponse {
  total: number;
  count: number;
  data: Question[];
}

// ── Constants ─────────────────────────────────────────────────────────────────

const CATEGORIES = [
  { id: "causality", label: "Causality" },
  { id: "construct", label: "Construct" },
  { id: "relation",  label: "Relation"  },
] as const;

/** Color sets per category — tag (inactive) vs active (selected) */
const CATEGORY_COLORS: Record<string, { tag: string; active: string; bar: string }> = {
  causality: {
    tag:    "bg-orange-100 dark:bg-orange-900/30 text-orange-800 dark:text-orange-300 border-orange-200 dark:border-orange-700",
    active: "bg-orange-500 text-white border-orange-500",
    bar:    "bg-orange-400",
  },
  construct: {
    tag:    "bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300 border-blue-200 dark:border-blue-700",
    active: "bg-blue-500 text-white border-blue-500",
    bar:    "bg-blue-400",
  },
  relation: {
    tag:    "bg-purple-100 dark:bg-purple-900/30 text-purple-800 dark:text-purple-300 border-purple-200 dark:border-purple-700",
    active: "bg-purple-500 text-white border-purple-500",
    bar:    "bg-purple-400",
  },
};

const QUESTION_TYPES_BY_CATEGORY: Record<string, { value: string; label: string }[]> = {
  construct: [
    { value: "cognitive_scene",             label: "Cognitive Scene" },
    { value: "perspective_camera_matching", label: "Perspective Camera Matching" },
    { value: "ego_rotation",                label: "Ego Rotation" },
    { value: "camera_ordering",             label: "Camera Ordering" },
    { value: "leave_one_camera_out",        label: "Leave One Camera Out" },
    { value: "camera_memory",               label: "Camera Memory" },
  ],
  relation: [
    { value: "multi_steps_reasoning",      label: "Multi Steps Reasoning" },
    { value: "allocentric_imagination",    label: "Allocentric Imagination" },
    { value: "spatial_compability",        label: "Spatial Compatibility" },
    { value: "scene_counting",             label: "Scene Counting" },
    { value: "multi_view_object_matching", label: "Multi-View Object Matching" },
    { value: "multi_view_depth_awareness", label: "Multi-View Depth Awareness" },
    { value: "direction",                  label: "Direction" },
    { value: "distance",                   label: "Distance" },
    { value: "distance_absolute",          label: "Distance Absolute" },
    { value: "scene_counting_absolute",   label: "Scene Counting Absolute" },
  ],
  causality: [
    { value: "event_ordering",         label: "Event Ordering" },
    { value: "object_interaction",     label: "Object Interaction" },
    { value: "trajectory_reasoning",   label: "Trajectory Reasoning" },
    { value: "trajectory_prediction",  label: "Trajectory Prediction" },
    { value: "occlusion_awareness",    label: "Occlusion Awareness" },
    { value: "object_disappearance",   label: "Object Disappearance" },
    { value: "object_manipulation",    label: "Object Manipulation" },
    { value: "action_reasoning",       label: "Action Reasoning" },
    { value: "interaction_reasoning",  label: "Interaction Reasoning" },
  ],
};

// ── Component ─────────────────────────────────────────────────────────────────

export default function QuestionPage() {
  const router = useRouter();
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  // Question state
  const [currentQuestion, setCurrentQuestion] = useState<Question | null>(null);
  const [currentOffset, setCurrentOffset]     = useState<number>(0);
  const [totalQuestions, setTotalQuestions]   = useState<number>(0);
  const [totalUnfiltered, setTotalUnfiltered] = useState<number>(0); // grand total without filters
  const [loading, setLoading]                 = useState<boolean>(false);
  const [error, setError]                     = useState<string>("");

  // Filter state
  const [filterCategory, setFilterCategory] = useState<string>("");
  const [filterType, setFilterType]         = useState<string>("");
  const [filterSource, setFilterSource]     = useState<string>("");
  const [filterStatus, setFilterStatus]     = useState<string>("pending"); // default: show only pending

  // Available sources
  const [availableSources, setAvailableSources] = useState<{ source: string; count: number }[]>([]);

  // Metadata panel collapsed state
  const [metaOpen, setMetaOpen] = useState<boolean>(true);

  // Review state
  const [reviewing, setReviewing] = useState<boolean>(false);

  // Review timer — tracks seconds spent on current question
  const timerStartRef  = useRef<number>(Date.now());
  const [timerDisplay, setTimerDisplay] = useState<number>(0);

  // Tick every second
  useEffect(() => {
    const id = setInterval(() => {
      setTimerDisplay(Math.floor((Date.now() - timerStartRef.current) / 1000));
    }, 1000);
    return () => clearInterval(id);
  }, []);

  // Reset timer whenever a new question loads
  useEffect(() => {
    if (currentQuestion) {
      timerStartRef.current = Date.now();
      setTimerDisplay(0);
    }
  }, [currentQuestion?.question_id]); // eslint-disable-line react-hooks/exhaustive-deps

  // Username from localStorage (set during login)
  const [username, setUsername] = useState<string>("");
  useEffect(() => {
    try {
      const stored = localStorage.getItem("drivespatial_auth");
      if (stored) setUsername(JSON.parse(stored).username ?? "");
    } catch {
      // ignore
    }
  }, []);

  // ── Bootstrap ────────────────────────────────────────────────────────────────

  useEffect(() => {
    // Fetch sources list
    fetch(`${apiUrl}/api/v1/questions/sources/list`)
      .then((r) => r.json())
      .then((d) => setAvailableSources(d.sources || []))
      .catch((e) => console.error("Failed to load sources:", e));

    // Fetch grand total (no filters)
    fetch(`${apiUrl}/api/v1/questions?top_k=1&offset=0`)
      .then((r) => r.json())
      .then((d: QuestionResponse) => setTotalUnfiltered(d.total))
      .catch(() => {});
  }, [apiUrl]);

  // ── Data fetching ─────────────────────────────────────────────────────────────

  const loadQuestion = useCallback(
    async (
      offset: number,
      category: string = filterCategory,
      type: string     = filterType,
      source: string   = filterSource,
      status: string   = filterStatus
    ) => {
      try {
        setLoading(true);
        setError("");

        const params = new URLSearchParams({ top_k: "1", offset: String(offset) });
        if (category) params.set("question_category", category);
        if (type)     params.set("question_type", type);
        if (source)   params.set("source", source);
        if (status)   params.set("status", status);

        const response = await fetch(`${apiUrl}/api/v1/questions?${params}`);
        if (!response.ok) throw new Error(`Failed to load: ${response.statusText}`);

        const data: QuestionResponse = await response.json();

        if (data.count > 0) {
          setCurrentQuestion(data.data[0]);
          setTotalQuestions(data.total);
          setCurrentOffset(offset);
        } else {
          setCurrentQuestion(null);
          setTotalQuestions(0);
          setError("No questions found for the current filters.");
        }
      } catch (err) {
        console.error("Error loading question:", err);
        setError(err instanceof Error ? err.message : "Failed to load question");
      } finally {
        setLoading(false);
      }
    },
    [apiUrl, filterCategory, filterType, filterSource, filterStatus]
  );

  // Reset to page 0 and reload whenever any filter changes
  useEffect(() => {
    setCurrentOffset(0);
    loadQuestion(0, filterCategory, filterType, filterSource, filterStatus);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filterCategory, filterType, filterSource, filterStatus]);

  // ── Navigation ────────────────────────────────────────────────────────────────

  const handleNext = () => {
    if (currentOffset < totalQuestions - 1) loadQuestion(currentOffset + 1);
  };

  const handlePrevious = () => {
    if (currentOffset > 0) loadQuestion(currentOffset - 1);
  };

  // ── Review ────────────────────────────────────────────────────────────────────

  const handleReview = async (status: QuestionStatus) => {
    if (!currentQuestion) return;
    try {
      setReviewing(true);
      const timeSpent = parseFloat(
        ((Date.now() - timerStartRef.current) / 1000).toFixed(2)
      );
      const response = await fetch(
        `${apiUrl}/api/v1/questions/${currentQuestion.question_id}/status`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            status,
            reviewed_by: username || undefined,
            time_spent_review: timeSpent,
          }),
        }
      );
      if (!response.ok) throw new Error(`Server returned ${response.status}`);
      const updated: Question = await response.json();
      console.log(`[Review] Question ${updated.question_id} marked as ${status}`);

      // If a status filter is active and this question no longer matches it,
      // reload at the same offset so the next matching question slides in.
      // Otherwise just update in-place.
      if (filterStatus && updated.status !== filterStatus) {
        // Re-fetch at the same offset (next question that matches the filter)
        await loadQuestion(currentOffset);
      } else {
        // Update in-place so the status badge refreshes
        setCurrentQuestion(updated);
        // On accept with no status filter → auto-advance (ambiguous/reject do not)
        if (status === "accepted" && currentOffset < totalQuestions - 1) {
          await loadQuestion(currentOffset + 1);
        }
      }
    } catch (err) {
      console.error("Failed to update question status:", err);
    } finally {
      setReviewing(false);
    }
  };

  // ── Helpers ───────────────────────────────────────────────────────────────────

  const toggleCategory = (cat: string) => {
    const next = filterCategory === cat ? "" : cat;
    // Reset type if it doesn't belong to the new category
    if (next && filterType) {
      const valid = QUESTION_TYPES_BY_CATEGORY[next]?.map((t) => t.value) ?? [];
      if (!valid.includes(filterType)) setFilterType("");
    }
    setFilterCategory(next);
  };

  const clearFilters = () => {
    setFilterCategory("");
    setFilterType("");
    setFilterSource("");
    setFilterStatus("");
  };

  const isFiltered = !!(filterCategory || filterType || filterSource || filterStatus);
  const progressPct = totalQuestions > 0
    ? ((currentOffset + 1) / totalQuestions) * 100
    : 0;

  const activeBarColor =
    filterCategory
      ? CATEGORY_COLORS[filterCategory]?.bar ?? "bg-blue-400"
      : "bg-blue-400";

  const visibleTypes = filterCategory
    ? QUESTION_TYPES_BY_CATEGORY[filterCategory] ?? []
    : Object.values(QUESTION_TYPES_BY_CATEGORY).flat();

  // ── Render ────────────────────────────────────────────────────────────────────

  return (
    <div className="h-screen w-screen bg-zinc-50 dark:bg-zinc-950 font-sans flex overflow-hidden">



      {/* ══ MAIN CONTENT ══════════════════════════════════════════════════════════ */}
      <div className="flex-1 flex flex-col overflow-hidden">

        {/* ── Header ── */}
        <header className="bg-white dark:bg-zinc-900 border-b border-zinc-200 dark:border-zinc-800 flex-shrink-0">
          <div className="px-8 py-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <h1 className="text-xl font-bold text-zinc-900 dark:text-zinc-50">
                Question Answering
              </h1>
              {currentQuestion && (
                <div className="flex items-center gap-2">
                  <span
                    className={`px-2.5 py-0.5 rounded-full text-xs font-semibold border ${
                      CATEGORY_COLORS[currentQuestion.question_category]?.active ??
                      "bg-zinc-200 text-zinc-700 border-zinc-300"
                    }`}
                  >
                    {currentQuestion.question_category}
                  </span>
                  <span className="px-2.5 py-0.5 rounded-full text-xs font-medium bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 border border-zinc-200 dark:border-zinc-700">
                    {currentQuestion.question_type.replace(/_/g, " ")}
                  </span>
                </div>
              )}
            </div>

            {/* Header progress pill */}
            {totalQuestions > 0 && (
              <div className="flex items-center gap-3">
                <span className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">
                  {currentOffset + 1}
                  <span className="font-normal text-zinc-400 dark:text-zinc-500"> / {totalQuestions}</span>
                  {isFiltered && totalUnfiltered > 0 && (
                    <span className="text-xs text-zinc-400 dark:text-zinc-500 ml-1">
                      (filtered)
                    </span>
                  )}
                </span>
              </div>
            )}
          </div>

          {/* ── Full-width progress bar under header ── */}
          <div className="h-1 w-full bg-zinc-100 dark:bg-zinc-800">
            <div
              className={`h-full transition-all duration-300 ${
                loading ? "animate-pulse bg-zinc-300 dark:bg-zinc-600 w-full" : activeBarColor
              }`}
              style={loading ? undefined : { width: `${progressPct}%` }}
            />
          </div>
        </header>

        {/* ── Question Body — two-panel layout ── */}
        <div className="flex-1 flex overflow-hidden min-h-0">

          {/* Loading */}
          {loading && (
            <div className="flex-1 flex items-center justify-center">
              <div className="animate-spin rounded-full h-10 w-10 border-4 border-zinc-200 border-t-blue-500" />
            </div>
          )}

          {/* Error */}
          {!loading && error && (
            <div className="flex-1 flex items-center justify-center">
              <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-700 text-red-700 dark:text-red-400 px-6 py-4 rounded-lg text-sm max-w-md text-center">
                {error}
              </div>
            </div>
          )}

          {!loading && !error && currentQuestion && (
            <>
              {/* ── LEFT: Images / Frames panel ── */}
              {(() => {
                const isVideo   = currentQuestion.media_type === "video";
                const hasImages = !isVideo && currentQuestion.images && currentQuestion.images.length > 0;
                const hasFrames = isVideo && currentQuestion.frame_paths && currentQuestion.frame_paths.length > 0;

                if (!hasImages && !hasFrames) return null;

                const frameBase = isVideo
                  ? `data/questions-v8/${currentQuestion.question_category}/${currentQuestion.question_type}/images/${currentQuestion.question_id}`
                  : null;

                const displayItems: { path: string; label: string }[] = isVideo
                  ? (currentQuestion.frame_paths ?? []).map((fn) => ({
                      path:  `${frameBase}/${fn}`,
                      label: fn,
                    }))
                  : (currentQuestion.images ?? []).map((p) => ({
                      path:  p,
                      label: p.split("/").pop() ?? p,
                    }));

                // ── No-scroll layout: context on top, 2×2 options grid below ──────
                const GRID_TYPES = ["cognitive_scene", "perspective_camera_matching"];
                const isGridLayout = GRID_TYPES.includes(currentQuestion.question_type)
                  && displayItems.length >= 2;

                if (isGridLayout) {
                  const [context, ...options] = displayItems;
                  return (
                    <div className="flex-1 flex flex-col bg-zinc-950 border-r border-zinc-800 overflow-hidden min-w-0 min-h-0">

                      {/* Context image — top 55% */}
                      <div className="flex-[11] min-h-0 flex flex-col px-4 pt-4 pb-2 gap-1.5">
                        <ImageLabel index={1} filename={context.label} tag="Context" />
                        <div className="flex-1 min-h-0">
                          <HpcImage relativePath={context.path} alt="Context" fill />
                        </div>
                      </div>

                      <div className="border-t border-zinc-800 mx-4" />

                      {/* Option images — bottom 45%, 2×2 grid */}
                      <div className="flex-[9] min-h-0 grid grid-cols-2 grid-rows-2 gap-2 px-4 pb-4 pt-2">
                        {options.slice(0, 4).map((item, i) => {
                          const isCorrect = i === currentQuestion.correct_answer;
                          return (
                            <div
                              key={i}
                              className={`min-h-0 flex flex-col gap-1 rounded-xl ${
                                isCorrect
                                  ? "ring-2 ring-green-500 ring-offset-2 ring-offset-zinc-950"
                                  : ""
                              }`}
                            >
                              <div className="flex items-center gap-1.5 flex-shrink-0">
                                <ImageLabel
                                  index={i + 2}
                                  filename={item.label}
                                  tag={`Option ${String.fromCharCode(65 + i)}`}
                                />
                                {isCorrect && (
                                  <span className="ml-auto inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md bg-green-500 text-white text-[10px] font-bold flex-shrink-0">
                                    <svg className="w-2.5 h-2.5" fill="currentColor" viewBox="0 0 20 20">
                                      <path fillRule="evenodd" d="M16.707 5.293a1 1 0 00-1.414 0L8 12.586 4.707 9.293a1 1 0 00-1.414 1.414l4 4a1 1 0 001.414 0l8-8a1 1 0 000-1.414z" clipRule="evenodd" />
                                    </svg>
                                    Correct
                                  </span>
                                )}
                              </div>
                              <div className="flex-1 min-h-0">
                                <HpcImage
                                  relativePath={item.path}
                                  alt={`Option ${String.fromCharCode(65 + i)}`}
                                  fill
                                  correct={isCorrect}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  );
                }

                // ── Default: vertical scroll list ────────────────────────────────
                return (
                  <div className="flex-1 overflow-y-auto bg-zinc-950 border-r border-zinc-800 p-5 space-y-4 min-w-0">
                    {/* Media type badge */}
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
                    </div>

                    {displayItems.map((item, i) => (
                      <div key={i} className="space-y-1.5">
                        <ImageLabel index={i + 1} filename={item.label} />
                        <HpcImage relativePath={item.path} alt={`Image ${i + 1}`} />
                      </div>
                    ))}
                  </div>
                );
              })()}

              {/* ── RIGHT: Text + Options panel ── */}
              <div className="w-[420px] flex-shrink-0 overflow-y-auto p-6 space-y-4 bg-zinc-50 dark:bg-zinc-950">

                {/* Status banner */}
                {currentQuestion.status !== "pending" && (
                  <div
                    className={`flex items-center gap-3 px-4 py-2.5 rounded-xl border text-sm font-medium ${
                      currentQuestion.status === "accepted"
                        ? "bg-green-50 dark:bg-green-900/20 border-green-300 dark:border-green-700 text-green-700 dark:text-green-400"
                        : currentQuestion.status === "ambiguous"
                        ? "bg-yellow-50 dark:bg-yellow-900/20 border-yellow-300 dark:border-yellow-700 text-yellow-700 dark:text-yellow-400"
                        : "bg-red-50 dark:bg-red-900/20 border-red-300 dark:border-red-700 text-red-700 dark:text-red-400"
                    }`}
                  >
                    {currentQuestion.status === "accepted" ? (
                      <svg className="w-4 h-4 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                    ) : currentQuestion.status === "ambiguous" ? (
                      <svg className="w-4 h-4 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                      </svg>
                    ) : (
                      <svg className="w-4 h-4 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                      </svg>
                    )}
                    <span>
                      Marked as <span className="font-bold">{currentQuestion.status}</span>
                    </span>
                    <button
                      onClick={() => handleReview("pending")}
                      className="ml-auto text-xs underline opacity-60 hover:opacity-100 transition-opacity"
                    >
                      Undo
                    </button>
                  </div>
                )}

                {/* Question text — image/video tokens replaced with numbered badges */}
                <div className="bg-white dark:bg-zinc-900 rounded-xl shadow-sm border border-zinc-200 dark:border-zinc-800 p-5">
                  <QuestionTextWithRefs
                    text={currentQuestion.question_text}
                    mediaType={currentQuestion.media_type}
                    imageCount={currentQuestion.images?.length ?? 0}
                    frameCount={currentQuestion.frame_paths?.length ?? 0}
                  />
                </div>

                {/* Text options */}
                {currentQuestion.options && currentQuestion.options.length > 0 && (
                  <div className="bg-white dark:bg-zinc-900 rounded-xl shadow-sm border border-zinc-200 dark:border-zinc-800 p-5">
                    <h3 className="text-xs font-semibold text-zinc-400 dark:text-zinc-500 uppercase tracking-wider mb-3">
                      Options
                    </h3>
                    <div className="space-y-2">
                      {currentQuestion.options.map((option, idx) => {
                        const isCorrect = idx === currentQuestion.correct_answer;
                        return (
                          <div
                            key={idx}
                            className={`flex items-start gap-3 p-3 rounded-lg border-2 ${
                              isCorrect
                                ? "border-green-400 bg-green-50 dark:bg-green-900/20"
                                : "border-zinc-200 dark:border-zinc-700 bg-zinc-50 dark:bg-zinc-800/50"
                            }`}
                          >
                            <div
                              className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                                isCorrect
                                  ? "bg-green-500 text-white"
                                  : "bg-zinc-200 dark:bg-zinc-700 text-zinc-600 dark:text-zinc-300"
                              }`}
                            >
                              {String.fromCharCode(65 + idx)}
                            </div>
                            <div className="flex-1 text-sm text-zinc-800 dark:text-zinc-200 pt-0.5">
                              {option}
                            </div>
                            {isCorrect && (
                              <svg className="w-4 h-4 text-green-500 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                              </svg>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}
              </div>
            </>
          )}
        </div>

        {/* ── Navigation + Review Footer ── */}
        <footer className="bg-white dark:bg-zinc-900 border-t border-zinc-200 dark:border-zinc-800 px-8 py-4 flex items-center gap-4 flex-shrink-0">

          {/* Accept / Ambiguous / Reject */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => handleReview("accepted")}
              disabled={reviewing || loading || !currentQuestion}
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-green-500 hover:bg-green-600 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-semibold rounded-lg transition-colors shadow-sm"
            >
              {reviewing ? (
                <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
              ) : (
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M16.707 5.293a1 1 0 00-1.414 0L8 12.586 4.707 9.293a1 1 0 00-1.414 1.414l4 4a1 1 0 001.414 0l8-8a1 1 0 000-1.414z" clipRule="evenodd" />
                </svg>
              )}
              Accept
            </button>

            <button
              onClick={() => handleReview("ambiguous")}
              disabled={reviewing || loading || !currentQuestion}
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-yellow-500 hover:bg-yellow-600 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-semibold rounded-lg transition-colors shadow-sm"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              Ambiguous
            </button>

            <button
              onClick={() => handleReview("rejected")}
              disabled={reviewing || loading || !currentQuestion}
              className="inline-flex items-center gap-2 px-5 py-2.5 bg-red-500 hover:bg-red-600 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-semibold rounded-lg transition-colors shadow-sm"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
              </svg>
              Reject
            </button>
          </div>

          {/* Timer */}
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700">
            <svg className="w-3.5 h-3.5 text-zinc-400 dark:text-zinc-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span className="text-sm font-mono font-bold text-zinc-700 dark:text-zinc-200 tabular-nums">
              {String(Math.floor(timerDisplay / 60)).padStart(2, "0")}:{String(timerDisplay % 60).padStart(2, "0")}
            </span>
          </div>

          {/* Divider */}
          <div className="w-px h-8 bg-zinc-200 dark:bg-zinc-700 mx-1" />

          {/* Dot stepper (centre-ish) */}
          <div className="flex-1 flex items-center justify-center gap-1.5">
            {Array.from({ length: Math.min(totalQuestions, 7) }, (_, i) => {
              const dotIdx    = Math.min(Math.floor((i / 6) * (totalQuestions - 1)), totalQuestions - 1);
              const isCurrent = dotIdx === currentOffset;
              return (
                <div
                  key={i}
                  className={`rounded-full transition-all ${
                    isCurrent
                      ? `w-3 h-3 ${activeBarColor}`
                      : "w-1.5 h-1.5 bg-zinc-300 dark:bg-zinc-600"
                  }`}
                />
              );
            })}
          </div>

          {/* Prev / Next */}
          <div className="flex items-center gap-2">
            <button
              onClick={handlePrevious}
              disabled={currentOffset === 0 || loading}
              className="px-4 py-2.5 bg-zinc-100 dark:bg-zinc-800 hover:bg-zinc-200 dark:hover:bg-zinc-700 disabled:opacity-40 disabled:cursor-not-allowed text-zinc-800 dark:text-zinc-200 text-sm font-medium rounded-lg transition-colors"
            >
              ← Prev
            </button>
            <button
              onClick={handleNext}
              disabled={currentOffset >= totalQuestions - 1 || loading}
              className="px-4 py-2.5 bg-zinc-700 dark:bg-zinc-600 hover:bg-zinc-800 dark:hover:bg-zinc-500 disabled:opacity-40 disabled:cursor-not-allowed text-white text-sm font-medium rounded-lg transition-colors"
            >
              Next →
            </button>
          </div>
        </footer>
      </div>

            {/* ══ LEFT SIDEBAR ══════════════════════════════════════════════════════════ */}
            <aside className="w-72 flex-shrink-0 bg-white dark:bg-zinc-900 border-r border-zinc-200 dark:border-zinc-800 flex flex-col overflow-hidden">

{/* Sidebar header */}
<div className="px-5 pt-5 pb-4 border-b border-zinc-200 dark:border-zinc-800 flex-shrink-0">
  <button
    onClick={() => router.push("/mode-select")}
    className="text-xs text-zinc-400 dark:text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-200 transition-colors mb-3 flex items-center gap-1"
  >
    ← Mode Select
  </button>
  <h2 className="text-xs font-bold text-zinc-500 dark:text-zinc-400 uppercase tracking-widest">
    Filters
  </h2>
</div>

{/* Scrollable filter body */}
<div className="flex-1 overflow-y-auto p-5 space-y-6 min-h-0">

  {/* ── Category tags ── */}
  <section>
    <p className="text-xs font-semibold text-zinc-400 dark:text-zinc-500 uppercase tracking-wider mb-2.5">
      Category
    </p>
    <div className="flex flex-wrap gap-2">
      {CATEGORIES.map(({ id, label }) => {
        const active = filterCategory === id;
        const c      = CATEGORY_COLORS[id];
        return (
          <button
            key={id}
            onClick={() => toggleCategory(id)}
            className={`px-3 py-1.5 rounded-full text-xs font-semibold border transition-all ${
              active ? c.active : c.tag
            }`}
          >
            {label}
          </button>
        );
      })}
    </div>
  </section>

  {/* ── Question type ── */}
  
  <section>
    <p className="text-xs font-semibold text-zinc-400 dark:text-zinc-500 uppercase tracking-wider mb-2.5">
      Question Type
    </p>
    <select
      value={filterType}
      onChange={(e) => setFilterType(e.target.value)}
      className="w-full px-3 py-2 text-xs bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-lg text-zinc-700 dark:text-zinc-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
    >
      <option value="">All Types</option>
      {filterCategory ? (
        visibleTypes.map((t) => (
          <option key={t.value} value={t.value}>{t.label}</option>
        ))
      ) : (
        Object.entries(QUESTION_TYPES_BY_CATEGORY).map(([cat, types]) => (
          <optgroup key={cat} label={cat.charAt(0).toUpperCase() + cat.slice(1)}>
            {types.map((t) => (
              <option key={t.value} value={t.value}>{t.label}</option>
            ))}
          </optgroup>
        ))
      )}
    </select>
  </section>

  {/* ── Source ── */}
  <section>
    <p className="text-xs font-semibold text-zinc-400 dark:text-zinc-500 uppercase tracking-wider mb-2.5">
      Source
    </p>
    <select
      value={filterSource}
      onChange={(e) => setFilterSource(e.target.value)}
      className="w-full px-3 py-2 text-xs bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-lg text-zinc-700 dark:text-zinc-200 focus:outline-none focus:ring-2 focus:ring-blue-500"
    >
      <option value="">All Sources</option>
      {availableSources.map(({ source, count }) => (
        <option key={source} value={source}>{source} ({count})</option>
      ))}
    </select>
  </section>

  {/* ── Status ── */}
  <section>
    <p className="text-xs font-semibold text-zinc-400 dark:text-zinc-500 uppercase tracking-wider mb-2.5">
      Status
    </p>
    <div className="flex flex-col gap-1.5">
      {[
        { value: "pending",   label: "○  Pending",   style: "text-zinc-600 dark:text-zinc-300" },
        { value: "accepted",  label: "✓  Accepted",  style: "text-green-700 dark:text-green-400" },
        { value: "ambiguous", label: "◐  Ambiguous", style: "text-yellow-700 dark:text-yellow-400" },
        { value: "rejected",  label: "✗  Rejected",  style: "text-red-700 dark:text-red-400" },
        { value: "",          label: "All statuses", style: "text-zinc-500 dark:text-zinc-400" },
      ].map(({ value, label, style }) => (
        <button
          key={value}
          onClick={() => setFilterStatus(value)}
          className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium border transition-all text-left ${
            filterStatus === value
              ? "bg-zinc-800 dark:bg-zinc-100 text-white dark:text-zinc-900 border-zinc-800 dark:border-zinc-100"
              : `bg-zinc-50 dark:bg-zinc-800 border-zinc-200 dark:border-zinc-700 hover:border-zinc-400 dark:hover:border-zinc-500 ${style}`
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  </section>

  {/* ── Active filter chips ── */}
  {isFiltered && (
    <section>
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs font-semibold text-zinc-400 dark:text-zinc-500 uppercase tracking-wider">
          Active
        </p>
        <button
          onClick={clearFilters}
          className="text-xs text-red-500 hover:text-red-700 dark:hover:text-red-400 transition-colors"
        >
          Clear all
        </button>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {filterCategory && (
          <Chip label={filterCategory} onRemove={() => setFilterCategory("")} />
        )}
        {filterType && (
          <Chip label={filterType.replace(/_/g, " ")} onRemove={() => setFilterType("")} />
        )}
                {filterSource && (
                  <Chip label={filterSource} onRemove={() => setFilterSource("")} />
                )}
                {filterStatus && (
                  <Chip label={filterStatus} onRemove={() => setFilterStatus("")} />
                )}
      </div>
    </section>
  )}

  {/* ── Metadata ── */}
  {currentQuestion && (
    <section>
      <button
        onClick={() => setMetaOpen((v) => !v)}
        className="w-full flex items-center justify-between mb-2.5"
      >
        <p className="text-xs font-semibold text-zinc-400 dark:text-zinc-500 uppercase tracking-wider">
          Metadata
        </p>
        <span className="text-zinc-400 text-xs">{metaOpen ? "▲" : "▼"}</span>
      </button>
              {metaOpen && (
                <div className="space-y-2.5 text-xs">
                  {/* Status */}
                  <div className="flex flex-col gap-0.5">
                    <span className="text-zinc-400 dark:text-zinc-500 capitalize">Status</span>
                    <StatusBadge status={currentQuestion.status} />
                  </div>

                  {/* Time spent reviewing */}
                  <div className="flex flex-col gap-0.5">
                    <span className="text-zinc-400 dark:text-zinc-500 capitalize">Time spent</span>
                    <span className="text-zinc-700 dark:text-zinc-300 font-mono text-xs">
                      {currentQuestion.time_spent_review > 0
                        ? `${Math.floor(currentQuestion.time_spent_review / 60)}m ${Math.round(currentQuestion.time_spent_review % 60)}s`
                        : "—"}
                    </span>
                  </div>

                  <MetaRow label="ID"          value={String(currentQuestion.id)} />
                  <MetaRow label="Question ID" value={currentQuestion.question_id} mono truncate />
                  <MetaRow label="Category"    value={currentQuestion.question_category} />
                  <MetaRow label="Type"        value={currentQuestion.question_type.replace(/_/g, " ")} />
          <MetaRow label="Media"       value={currentQuestion.media_type} />
          {currentQuestion.source   && <MetaRow label="Source"   value={currentQuestion.source} />}
          {currentQuestion.scene_id && <MetaRow label="Scene ID" value={currentQuestion.scene_id} mono truncate />}
          {currentQuestion.metadata?.record_id && (
            <MetaRow label="Record ID" value={String(currentQuestion.metadata.record_id)} />
          )}
          <MetaRow
            label="Created"
            value={new Date(currentQuestion.created_at).toLocaleDateString()}
          />
          {currentQuestion.metadata &&
            Object.entries(currentQuestion.metadata)
              .filter(([k]) => k !== "record_id")
              .map(([k, v]) => (
                <MetaRow
                  key={k}
                  label={k.replace(/_/g, " ")}
                  value={typeof v === "object" ? JSON.stringify(v) : String(v)}
                  truncate
                />
              ))}
        </div>
      )}
    </section>
  )}
</div>

{/* ── Sidebar progress block ── */}
<div className="flex-shrink-0 border-t border-zinc-200 dark:border-zinc-800 px-5 py-4 space-y-2.5">
  {/* Numbers */}
  <div className="flex items-end justify-between">
    <div>
      <span className="text-2xl font-bold text-zinc-800 dark:text-zinc-100 leading-none">
        {totalQuestions > 0 ? currentOffset + 1 : 0}
      </span>
      <span className="text-sm text-zinc-400 dark:text-zinc-500 ml-1">
        / {totalQuestions}
      </span>
    </div>
    {isFiltered && totalUnfiltered > 0 && (
      <span className="text-xs text-zinc-400 dark:text-zinc-500">
        of {totalUnfiltered.toLocaleString()} total
      </span>
    )}
  </div>

  {/* Progress bar */}
  <div className="w-full h-1.5 bg-zinc-200 dark:bg-zinc-700 rounded-full overflow-hidden">
    <div
      className={`h-full rounded-full transition-all duration-300 ${activeBarColor}`}
      style={{ width: `${progressPct}%` }}
    />
  </div>

  {/* Filter context label */}
  <p className="text-xs text-zinc-400 dark:text-zinc-500">
    {isFiltered ? "Filtered results" : "All questions"}
  </p>
</div>
</aside>
    </div>
  );
}

// ── Question text renderer (right panel) ──────────────────────────────────────

/**
 * Renders question_text with <image> and <video> tokens replaced by
 * small numbered badges referencing the media shown in the left panel.
 */
function QuestionTextWithRefs({
  text,
  mediaType,
  imageCount,
  frameCount,
}: {
  text: string;
  mediaType: string;
  imageCount: number;
  frameCount: number;
}) {
  const isVideo = mediaType === "video";
  // Split on whichever token the text uses
  const TOKEN = isVideo ? "<video>" : "<image>";
  const totalItems = isVideo ? frameCount : imageCount;
  const parts = text.split(TOKEN);

  return (
    <div className="text-sm leading-relaxed text-zinc-800 dark:text-zinc-200">
      {parts.map((part, idx) => (
        <Fragment key={idx}>
          {part && <span className="whitespace-pre-wrap">{part}</span>}
          {idx < parts.length - 1 && (
            isVideo ? (
              // Video token → show a single "frames 1–N" badge
              <span className="inline-flex items-center gap-1 px-2 py-0.5 mx-0.5 rounded-md bg-violet-100 dark:bg-violet-900/40 text-violet-700 dark:text-violet-300 text-xs font-semibold align-middle">
                <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M2 6a2 2 0 012-2h6a2 2 0 012 2v8a2 2 0 01-2 2H4a2 2 0 01-2-2V6zm12.553 1.106A1 1 0 0014 8v4a1 1 0 00.553.894l2 1A1 1 0 0018 13V7a1 1 0 00-1.447-.894l-2 1z" />
                </svg>
                frames 1–{totalItems}
              </span>
            ) : (
              // Image token → numbered badge per image
              idx < totalItems && (
                <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 mx-0.5 rounded-md bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 text-xs font-mono font-semibold align-middle">
                  ⬡ {idx + 1}
                </span>
              )
            )
          )}
        </Fragment>
      ))}
    </div>
  );
}

// ── HPC Image component (left panel) ──────────────────────────────────────────

/** Single HPC image with loading / error states, sized to fill the left panel. */
function HpcImage({
  relativePath,
  alt,
  fill = false,
  correct = false,
}: {
  relativePath: string | null;
  alt: string;
  /** If true, the image stretches to fill its container height (use inside a sized flex/grid cell). */
  fill?: boolean;
  /** If true, adds a green tint overlay to mark this as the correct option. */
  correct?: boolean;
}) {
  const [status, setStatus] = useState<"loading" | "loaded" | "error">("loading");

  useEffect(() => {
    setStatus("loading");
  }, [relativePath]);

  const wrapCls = fill
    ? "w-full h-full rounded-xl overflow-hidden bg-zinc-900 border border-zinc-800"
    : "rounded-xl overflow-hidden bg-zinc-900 border border-zinc-800";

  if (!relativePath) {
    return (
      <div className={`${wrapCls} flex items-center justify-center text-xs text-zinc-500`}>
        [missing image]
      </div>
    );
  }

  const src = hpcImageUrl(relativePath);

  return (
    <div className={wrapCls}>
      {status === "loading" && (
        <div className={`flex items-center justify-center ${fill ? "h-full" : "h-40"}`}>
          <div className="animate-spin rounded-full h-7 w-7 border-[3px] border-zinc-700 border-t-blue-500" />
        </div>
      )}
      {status === "error" && (
        <div className={`flex flex-col items-center justify-center gap-1.5 text-xs text-zinc-500 ${fill ? "h-full" : "h-32"}`}>
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
              d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
          </svg>
          Failed to load image
        </div>
      )}
      <div className={`relative ${fill ? "h-full" : ""} ${status === "loaded" ? "block" : "hidden"}`}>
        <img
          src={src}
          alt={alt}
          className={`w-full object-contain rounded-xl ${fill ? "h-full" : ""}`}
          onLoad={() => {
            console.log("Loading image: ", src);
            setStatus("loaded")
          }}
          onError={() => setStatus("error")}
        />
        {correct && status === "loaded" && (
          <div className="absolute inset-0 rounded-xl bg-green-500/10 pointer-events-none" />
        )}
      </div>
    </div>
  );
}

// ── Small helper components ────────────────────────────────────────────────────

function ImageLabel({
  index,
  filename,
  tag,
}: {
  index: number;
  filename: string;
  tag?: string;
}) {
  return (
    <div className="flex items-center gap-1.5 flex-shrink-0">
      <span className="w-4 h-4 rounded-full bg-zinc-700 flex items-center justify-center text-zinc-300 font-bold text-[9px]">
        {index}
      </span>
      {tag && (
        <span className="text-[10px] font-semibold text-zinc-400 uppercase tracking-wide">
          {tag}
        </span>
      )}
      <span className="text-[10px] font-mono text-zinc-600 truncate">{filename}</span>
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const styles: Record<string, string> = {
    pending:   "bg-zinc-100 dark:bg-zinc-800 text-zinc-500 dark:text-zinc-400 border-zinc-200 dark:border-zinc-700",
    accepted:  "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 border-green-300 dark:border-green-700",
    ambiguous: "bg-yellow-100 dark:bg-yellow-900/30 text-yellow-700 dark:text-yellow-400 border-yellow-300 dark:border-yellow-700",
    rejected:  "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 border-red-300 dark:border-red-700",
  };
  const icons: Record<string, string> = {
    pending:   "○",
    accepted:  "✓",
    ambiguous: "◐",
    rejected:  "✗",
  };
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-xs font-semibold capitalize ${styles[status] ?? styles.pending}`}>
      <span>{icons[status] ?? "○"}</span>
      {status}
    </span>
  );
}

function Chip({ label, onRemove }: { label: string; onRemove: () => void }) {
  return (
    <span className="inline-flex items-center gap-1 px-2 py-1 bg-zinc-100 dark:bg-zinc-800 rounded text-xs text-zinc-700 dark:text-zinc-300 border border-zinc-200 dark:border-zinc-700">
      {label}
      <button onClick={onRemove} className="hover:text-red-500 leading-none">×</button>
    </span>
  );
}

function MetaRow({
  label,
  value,
  mono     = false,
  truncate = false,
}: {
  label: string;
  value: string;
  mono?: boolean;
  truncate?: boolean;
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-zinc-400 dark:text-zinc-500 capitalize text-xs">{label}</span>
      <span
        title={value}
        className={`text-zinc-700 dark:text-zinc-300 text-xs ${mono ? "font-mono" : ""} ${
          truncate ? "truncate" : "break-words"
        }`}
      >
        {value}
      </span>
    </div>
  );
}
