"use client";

import { useState, useEffect, useRef, memo, MutableRefObject } from "react";

interface TimerProps {
  checkMode: string;
  recordId: number | null;
  timerStartTimeRef: MutableRefObject<number>;
  onTimeLog?: (recordId: number, mode: string, timeSpent: number) => void;
}

/**
 * Isolated Timer Component
 * This component is memoized and isolated to prevent parent re-renders
 */
const Timer = memo(({ checkMode, recordId, timerStartTimeRef, onTimeLog }: TimerProps) => {
  const [displayTime, setDisplayTime] = useState<number>(0);
  const prevRecordIdRef = useRef<number | null>(recordId);
  const prevModeRef = useRef<string>(checkMode);

  // Update display time every second (isolated to this component only)
  useEffect(() => {
    const interval = setInterval(() => {
      const elapsed = (Date.now() - timerStartTimeRef.current) / 1000;
      setDisplayTime(elapsed);
    }, 1000);

    return () => clearInterval(interval);
  }, []); // Empty deps - never re-creates

  // Handle mode/record changes
  useEffect(() => {
    const getElapsedTime = () => {
      return (Date.now() - timerStartTimeRef.current) / 1000;
    };

    // Log time when record or mode changes (but not on initial mount)
    if (prevRecordIdRef.current !== null && recordId !== null) {
      const recordChanged = prevRecordIdRef.current !== recordId;
      const modeChanged = prevModeRef.current !== checkMode;

      if (recordChanged || modeChanged) {
        const timeSpent = getElapsedTime();
        if (timeSpent > 0.5 && onTimeLog) {
          // Log to the previous mode/record
          const modeToLog = modeChanged ? prevModeRef.current : checkMode;
          onTimeLog(prevRecordIdRef.current, modeToLog, timeSpent);
        }

        // Reset timer
        timerStartTimeRef.current = Date.now();
        setDisplayTime(0);
      }
    }

    // Update tracking refs
    prevRecordIdRef.current = recordId;
    prevModeRef.current = checkMode;
  }, [checkMode, recordId, onTimeLog, timerStartTimeRef]);

  return (
    <div className="mt-3 px-4 py-2 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
      <div className="flex items-center justify-between">
        <span className="text-sm font-mono font-bold text-white-900 dark:text-white-900">
          Counter:
        </span>
        <span className="text-sm font-mono font-bold text-blue-900 dark:text-blue-100">
          {Math.floor(displayTime / 60)}:{(Math.floor(displayTime) % 60).toString().padStart(2, '0')}
        </span>
      </div>
    </div>
  );
});

Timer.displayName = "Timer";

export default Timer;
