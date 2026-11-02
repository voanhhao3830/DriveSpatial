"use client";

import { useState, useEffect, useRef, Suspense } from "react";
import dynamic from "next/dynamic";
import { useRouter } from "next/navigation";
import CameraWithBbox from "../components/CameraWithBbox";
import Timer from "../components/Timer";

// Dynamically import LidarViewer with no SSR to avoid Three.js SSR issues
const LidarViewer = dynamic(() => import("../components/LidarViewer"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center bg-zinc-900 rounded-lg">
      <div className="text-white">Loading 3D viewer...</div>
    </div>
  ),
});

/**
 * Type definitions for AV data from the API
 */
interface AVDataRecord {
  id: number;
  scene_id: string;
  frame_id: string;
  source: string;
  views: {
    // nuScenes cameras
    CAM_FRONT?: string;
    CAM_FRONT_RIGHT?: string;
    CAM_FRONT_LEFT?: string;
    CAM_BACK?: string;
    CAM_BACK_LEFT?: string;
    CAM_BACK_RIGHT?: string;
    // TruckScenes cameras
    CAMERA_LEFT_FRONT?: string;
    CAMERA_LEFT_BACK?: string;
    CAMERA_RIGHT_FRONT?: string;
    CAMERA_RIGHT_BACK?: string;
    // Waymo cameras
    FRONT?: string;
    FRONT_LEFT?: string;
    FRONT_RIGHT?: string;
    SIDE_LEFT?: string;
    SIDE_RIGHT?: string;
    // Allow any other camera names
    [key: string]: string | undefined;
  };
  lidar: {
    path?: string;
    [key: string]: any;
  };
  annotation: {
    bbox_3d?: number[][];
    label_3d?: string[];
    track_3d?: string[];
    bbox_2d?: any;
    label_2d?: any;
    track_2d?: any;
    velocity?: any;
  };
  calibration?: {
    ego_pose?: number[][];
    [key: string]: any;
  };
  metadata: Record<string, any>;
  updated_metadata?: Record<string, any>;
  verified_metadata: number;
  verified_spatial_relations?: number;
  verified_internal_interactions?: number;
  verified_external_interactions?: number;
  spatial_relations?: Array<{
    source_object: { track_id: string; label: string };
    target_object: { track_id: string; label: string };
    relations: string[];
    rel_coords_in_source_frame: { forward: number; right: number; up: number };
    source_forward_vec_xy: { fx: number; fy: number };
  }>;
  updated_spatial_relations?: Array<{
    source_object: { track_id: string; label: string };
    target_object: { track_id: string; label: string };
    relations: string[];
    rel_coords_in_source_frame: { forward: number; right: number; up: number };
    source_forward_vec_xy: { fx: number; fy: number };
  }>;
  internal_interactions?: any[];
  external_interactions?: any[];
  occlusions?: Array<{
    source_object: { track_id: string; label: string };
    target_object: { track_id: string; label: string };
    visibility: number;
  }>;
  updated_internal_interactions?: any[];
  updated_external_interactions?: any[];
  updated_occlusions?: Array<{
    source_object: { track_id: string; label: string };
    target_object: { track_id: string; label: string };
    visibility: number;
  }>;
  verify_by?: string;
  created_at: string;
}

interface APIResponse {
  total: number;
  count: number;
  data: AVDataRecord[];
}

/**
 * Main page component for DriveSpatial AV Data Viewer
 * Displays camera images from autonomous vehicle dataset
 */
export default function GraphAnnotationPage() {
  const router = useRouter();
  
  // Authentication states
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(false);
  const [username, setUsername] = useState<string>("");
  const [authChecking, setAuthChecking] = useState<boolean>(true);

  // Load authentication from localStorage on mount
  useEffect(() => {
    const storedAuth = localStorage.getItem("drivespatial_auth");
    if (storedAuth) {
      try {
        const { username: storedUsername, timestamp } = JSON.parse(storedAuth);
        // Check if auth is less than 7 days old
        const sevenDaysInMs = 7 * 24 * 60 * 60 * 1000;
        if (Date.now() - timestamp < sevenDaysInMs && storedUsername) {
          setUsername(storedUsername);
          setIsAuthenticated(true);
          console.log(`Authenticated as: ${storedUsername}`);
        } else {
          // Clear expired auth and redirect to login
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
    setAuthChecking(false);
  }, [router]);

  const [currentRecord, setCurrentRecord] = useState<AVDataRecord | null>(null);
  const [currentOffset, setCurrentOffset] = useState<number>(0); // Global offset across all frames
  const [totalRecords, setTotalRecords] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>("");
  const [apiUrl, setApiUrl] = useState<string>(process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000");
  const [showLidar, setShowLidar] = useState<boolean>(true);
  const [viewMode, setViewMode] = useState<"grid" | "split">("split");
  
  // Filter states - default to unverified only
  const [checkMode, setCheckMode] = useState<"metadata" | "spatial" | "internal" | "external" | "occlusion" | "all">("metadata");
  const [filterSceneId, setFilterSceneId] = useState<string>("");
  const [filterSource, setFilterSource] = useState<string[]>([]); // Multiple data sources
  const [selectedTrackId, setSelectedTrackId] = useState<string>("all");
  const [availableTrackIds, setAvailableTrackIds] = useState<string[]>([]);

  // Timer ref - shared with Timer component
  const timerStartTimeRef = useRef<number>(Date.now());

  // Metadata editing states
  const [editingMetadata, setEditingMetadata] = useState(false);
  const [metadataWeather, setMetadataWeather] = useState<string>("");
  const [metadataSceneType, setMetadataSceneType] = useState<string>("");
  const [metadataDaytime, setMetadataDaytime] = useState<string>("");
  const [savingMetadata, setSavingMetadata] = useState(false);

  // Spatial relations editing states
  const [editingRelations, setEditingRelations] = useState(false);
  const [relationEdits, setRelationEdits] = useState<Map<string, string[]>>(new Map()); // key: "source_target", value: relations array
  const [editingExternalInteractions, setEditingExternalInteractions] = useState(false);
  const [externalInteractionEdits, setExternalInteractionEdits] = useState<Map<string, string>>(new Map()); // key: "source_target", value: interaction_type string
  const [savingRelations, setSavingRelations] = useState(false);
  const [verifyingFrame, setVerifyingFrame] = useState(false);
  const [markingDone, setMarkingDone] = useState(false);

  // Cache for prefetched data
  const [dataCache, setDataCache] = useState<Map<number, AVDataRecord>>(new Map());

  // Camera configurations for different sources
  const getCameraConfig = (source: string, views?: Record<string, string | undefined>) => {
    switch (source?.toLowerCase()) {
      case "nuscenes":
        return {
          displayOrder: [
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT",
            "CAM_BACK",
            "CAM_BACK_RIGHT",
          ],
          apiOrder: [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
          ],
        };
      case "truckscenes":
        return {
          displayOrder: [
            "CAMERA_LEFT_FRONT",
            "CAMERA_RIGHT_FRONT",
            "CAMERA_LEFT_BACK",
            "CAMERA_RIGHT_BACK",
          ],
          apiOrder: [
            "CAMERA_LEFT_BACK",
            "CAMERA_LEFT_FRONT",
            "CAMERA_RIGHT_BACK",
            "CAMERA_RIGHT_FRONT",
          ],
        };
      case "waymo":
        return {
          displayOrder: [
            "FRONT_LEFT",
            "FRONT",
            "FRONT_RIGHT",
            "SIDE_LEFT",
            "SIDE_RIGHT",
          ],
          apiOrder: [
            "FRONT",
            "FRONT_LEFT",
            "FRONT_RIGHT",
            "SIDE_LEFT",
            "SIDE_RIGHT",
          ],
        };
      case "argoverse2":
      case "av2":
        return {
          displayOrder: [
            // Top row: Front cameras (left to right)
            "ring_front_left",
            "ring_front_center",
            "ring_front_right",
            // Bottom row: Side and rear cameras (left to right)
            "ring_side_left",
            "ring_rear_left",
            "ring_rear_right",
            "ring_side_right",
          ],
          apiOrder: [
            // API order matches data structure
            "ring_rear_left",
            "ring_side_left",
            "ring_front_left",
            "ring_front_center",
            "ring_front_right",
            "ring_side_right",
            "ring_rear_right",
          ],
        };
      default:
        // Generic fallback: auto-detect from available views
        if (views) {
          const availableCameras = Object.keys(views).filter(key => views[key]);
          return {
            displayOrder: availableCameras,
            apiOrder: availableCameras,
          };
        }
        return {
          displayOrder: [],
          apiOrder: [],
        };
    }
  };

  // Get current camera configuration based on current record source
  const currentCameraConfig = currentRecord
    ? getCameraConfig(currentRecord.source, currentRecord.views)
    : { displayOrder: [], apiOrder: [] };

  // Create mapping from display order to API order for current source
  const getCameraApiIndex = (cameraName: string): number => {
    return currentCameraConfig.apiOrder.indexOf(cameraName);
  };

  /**
   * Handle logout
   */
  const handleLogout = (): void => {
    // Clear auth from localStorage
    localStorage.removeItem("drivespatial_auth");
    console.log("User logged out");
    // Redirect to login
    router.push("/");
  };

  /**
   * Fetches data from the FastAPI backend
   * 
   * Three navigation modes (all instant using record ID):
   * 1. Direct ID: fetchData(recordId, 'direct', actualOffset)
   * 2. Next record: fetchData(currentRecordId, 'next')
   * 3. Previous record: fetchData(currentRecordId, 'prev')
   * 4. Initial load: fetchData(0, 'offset') - only used for first load
   */
  const fetchData = async (
    idOrOffset: number, 
    mode: 'direct' | 'next' | 'prev' | 'offset' = 'offset',
    actualOffset?: number,
    skipCache: boolean = false
  ): Promise<void> => {
    try {
      console.log(`[fetchData] Called with mode=${mode}, idOrOffset=${idOrOffset}, filterSource=`, filterSource, `skipCache=${skipCache}`);
      setLoading(true);
      setError("");

      // Check cache only for offset-based (initial load) and if not skipping cache
      if (mode === 'offset' && !skipCache) {
        const cachedData = dataCache.get(idOrOffset);
        if (cachedData) {
          console.log(`Using cached data for offset ${idOrOffset}`);
          setCurrentRecord(cachedData);
          setCurrentOffset(idOrOffset);
          
          const trackIds = ["00000", ...(cachedData.annotation?.track_3d || [])];
          setAvailableTrackIds(trackIds);
          setSelectedTrackId("all");
          
          setMetadataWeather(cachedData.updated_metadata?.weather || cachedData.metadata?.weather || "");
          setMetadataSceneType(cachedData.updated_metadata?.scene_type || cachedData.metadata?.scene_type || "");
          setMetadataDaytime(cachedData.updated_metadata?.daytime || cachedData.metadata?.daytime || "");
          setLoading(false);
          
          prefetchAdjacentFrames(idOrOffset);
          return;
        }
      }

      // Build query parameters based on navigation mode
      const params = new URLSearchParams({ top_k: "1" });

      if (mode === 'direct') {
        // Direct ID lookup (for scene jumps)
        params.append("record_id", idOrOffset.toString());
        console.log(`Fetching by direct ID: ${idOrOffset}, offset: ${actualOffset}`);
      } else if (mode === 'next') {
        // Next record after this ID
        params.append("next_id", idOrOffset.toString());
        console.log(`Fetching next record after ID: ${idOrOffset}`);
      } else if (mode === 'prev') {
        // Previous record before this ID
        params.append("prev_id", idOrOffset.toString());
        console.log(`Fetching previous record before ID: ${idOrOffset}`);
      } else {
        // Offset-based (only for initial load)
        params.append("offset", idOrOffset.toString());
      }

      // Add filters for all navigation types
      if (filterSceneId.trim()) {
        params.append("scene_id", filterSceneId.trim());
      }
      if (filterSource.length > 0) {
        console.log(`Adding source filters: ${filterSource.join(', ')}`);
        filterSource.forEach(source => {
          params.append("source", source);
        });
      }

      console.log(`Request URL: ${apiUrl}/api/v1/data?${params.toString()}`);
      const response = await fetch(`${apiUrl}/api/v1/data?${params.toString()}`);

      if (!response.ok) {
        throw new Error(`API request failed: ${response.statusText}`);
      }

      const data: APIResponse = await response.json();

      if (data.count === 0) {
        setError("No more records available");
        return;
      }

      const record = data.data[0];
      
      // Log spatial relations info for debugging
      console.log(`Record ${record.id} spatial relations:`, {
        has_spatial_relations: !!record.spatial_relations,
        spatial_relations_count: record.spatial_relations?.length || 0,
        has_updated_spatial_relations: !!record.updated_spatial_relations,
        updated_spatial_relations_count: record.updated_spatial_relations?.length || 0,
      });
      
      console.log(`Record ${record.id} internal interactions:`, {
        has_internal_interactions: !!record.internal_interactions,
        internal_interactions_count: record.internal_interactions?.length || 0,
        sample_interaction: record.internal_interactions?.[0],
        verified_internal: record.verified_internal_interactions,
        verify_by: record.verify_by,
      });
      
      setCurrentRecord(record);
      setTotalRecords(data.total);
      
      // Set the current offset based on mode
      if (mode === 'direct' && actualOffset !== undefined) {
        // Direct ID with known offset (from scene jump)
        console.log(`Loaded record ID ${record.id} at global offset ${actualOffset}`);
        setCurrentOffset(actualOffset);
      } else if (mode === 'next' || mode === 'prev') {
        // Next/prev navigation: calculate offset from current
        if (mode === 'next') {
          setCurrentOffset(currentOffset + 1);
          console.log(`Loaded next record ID ${record.id}, offset ${currentOffset + 1}`);
        } else {
          setCurrentOffset(currentOffset - 1);
          console.log(`Loaded previous record ID ${record.id}, offset ${currentOffset - 1}`);
        }
      } else {
        // Offset-based (initial load)
        setCurrentOffset(idOrOffset);
      }
      
      // Extract available track IDs from annotations
      const trackIds = ["00000", ...(record.annotation?.track_3d || [])];
      setAvailableTrackIds(trackIds);
      setSelectedTrackId("all");
      
      // Update metadata editing states
      setMetadataWeather(record.updated_metadata?.weather || record.metadata?.weather || "");
      setMetadataSceneType(record.updated_metadata?.scene_type || record.metadata?.scene_type || "");
      setMetadataDaytime(record.updated_metadata?.daytime || record.metadata?.daytime || "");

      // Cache current record by offset (only for offset-based)
      if (mode === 'offset') {
        setDataCache(prev => {
          const newCache = new Map(prev);
          newCache.set(idOrOffset, record);
          return newCache;
        });
        prefetchAdjacentFrames(idOrOffset, data.total);
      }
    } catch (err) {
      console.error("Error fetching data:", err);
      setError(err instanceof Error ? err.message : "Failed to fetch data");
    } finally {
      setLoading(false);
    }
  };

  /**
   * Prefetch adjacent frames in background (LIMITED to reduce lag)
   */
  const prefetchAdjacentFrames = async (currentOffset: number, total?: number): Promise<void> => {
    const totalRecordsCount = total || totalRecords;
    
    // Build base params
    const buildParams = (offset: number) => {
      const params = new URLSearchParams({
        top_k: "1",
        offset: offset.toString(),
      });

      if (filterSceneId.trim()) {
        params.append("scene_id", filterSceneId.trim());
      }

      return params;
    };

    // OPTIMIZED: Only prefetch immediate next frame to reduce lag
    const prefetchPromises: Promise<void>[] = [];

    // Prefetch ONLY next frame (not next 5)
    if (currentOffset < totalRecordsCount - 1 && !dataCache.has(currentOffset + 1)) {
      prefetchPromises.push(
        fetch(`${apiUrl}/api/v1/data?${buildParams(currentOffset + 1).toString()}`)
          .then(res => res.json())
          .then((data: APIResponse) => {
            if (data.count > 0) {
              setDataCache(prev => {
                const newCache = new Map(prev);
                newCache.set(currentOffset + 1, data.data[0]);
                return newCache;
              });
              console.log(`[Prefetch] Cached frame ${currentOffset + 1}`);
            }
          })
          .catch(err => console.debug(`Failed to prefetch next frame: ${err}`))
      );
    }

    // Execute prefetch (only 1 frame ahead for performance)
    if (prefetchPromises.length > 0) {
      await Promise.all(prefetchPromises);
    }
  };

  /**
   * Load initial data on component mount
   */
  useEffect(() => {
    console.log(`Filter changed - filterSource:`, filterSource, `filterSceneId:`, filterSceneId);
    // Clear cache when filters change and force fresh fetch
    setDataCache(new Map());
    fetchData(0, 'offset', undefined, true); // skipCache=true
  }, [checkMode, filterSceneId, filterSource]);

  /**
   * Log time spent to backend (called by Timer component)
   */
  const logTimeSpent = async (recordId: number, mode: string, timeSpent: number): Promise<void> => {
    try {
      const response = await fetch(
        `${apiUrl}/api/v1/data/${recordId}/log-time`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ 
            mode,
            time_spent: timeSpent 
          }),
        }
      );

      if (response.ok) {
        const result = await response.json();
        console.log(`[Timer] Logged ${timeSpent.toFixed(1)}s for ${mode} mode. Total: ${result.total_time.toFixed(1)}s`);
      }
    } catch (err) {
      console.error("Error logging time:", err);
      // Don't show error to user - this is a background operation
    }
  };

  /**
   * Handle source selection toggle
   */
  const handleSourceToggle = (source: string): void => {
    setFilterSource(prev => {
      const newSources = prev.includes(source)
        ? prev.filter(s => s !== source)
        : [...prev, source];
      console.log(`Source toggle: ${source}, new sources:`, newSources);
      return newSources;
    });
  };

  /**
   * Handle next record - uses record ID navigation (instant)
   */
  const handleNext = (): void => {
    if (currentRecord && currentRecord.id) {
      fetchData(currentRecord.id, 'next');
    }
  };

  /**
   * Handle previous record - uses record ID navigation (instant)
   */
  const handlePrevious = (): void => {
    if (currentRecord && currentRecord.id) {
      fetchData(currentRecord.id, 'prev');
    }
  };

  /**
   * Handle image load error
   */
  const handleImageError = (cameraName: string): void => {
    console.error(`Failed to load image for ${cameraName}`);
  };

  /**
   * Update metadata and mark as verified
   */
  const handleUpdateMetadata = async (): Promise<void> => {
    if (!currentRecord) return;

    try {
      // Log time spent before saving
      const timeSpent = (Date.now() - timerStartTimeRef.current) / 1000;
      if (timeSpent > 0.5) {
        await logTimeSpent(currentRecord.id, checkMode, timeSpent);
        timerStartTimeRef.current = Date.now(); // Reset timer
      }

      setSavingMetadata(true);
      setError("");

      const updateData = {
        updated_metadata: {
          weather: metadataWeather,
          scene_type: metadataSceneType,
          daytime: metadataDaytime,
        },
        verified_metadata: 1,
        verify_by: username,
      };

      const response = await fetch(
        `${apiUrl}/api/v1/data/${currentRecord.id}`,
        {
          method: "PUT",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(updateData),
        }
      );

      if (!response.ok) {
        throw new Error(`Failed to update metadata: ${response.statusText}`);
      }

      // Fetch the updated record directly by ID without changing offset
      const updatedResponse = await fetch(`${apiUrl}/api/v1/data/${currentRecord.id}`);
      if (updatedResponse.ok) {
        const updatedData = await updatedResponse.json();
        setCurrentRecord(updatedData);
        
        // Update editing states with new data
        setMetadataWeather(updatedData.updated_metadata?.weather || updatedData.metadata?.weather || "");
        setMetadataSceneType(updatedData.updated_metadata?.scene_type || updatedData.metadata?.scene_type || "");
        setMetadataDaytime(updatedData.updated_metadata?.daytime || updatedData.metadata?.daytime || "");
      }
      
      setEditingMetadata(false);
    } catch (err) {
      console.error("Error updating metadata:", err);
      setError(err instanceof Error ? err.message : "Failed to update metadata");
    } finally {
      setSavingMetadata(false);
    }
  };

  /**
   * Apply metadata to all frames in the current scene
   */
  const handleApplyToScene = async (): Promise<void> => {
    if (!currentRecord || !currentRecord.scene_id) return;

    try {
      // Log time spent before saving
      const timeSpent = (Date.now() - timerStartTimeRef.current) / 1000;
      if (timeSpent > 0.5) {
        await logTimeSpent(currentRecord.id, checkMode, timeSpent);
        timerStartTimeRef.current = Date.now(); // Reset timer
      }

      setSavingMetadata(true);
      setError("");

      const updateData = {
        updated_metadata: {
          weather: metadataWeather,
          scene_type: metadataSceneType,
          daytime: metadataDaytime,
        },
        verified_metadata: 1,
        verify_by: username,
      };

      // Fetch all frames in the current scene
      const sceneResponse = await fetch(
        `${apiUrl}/api/v1/data?scene_id=${encodeURIComponent(currentRecord.scene_id)}&top_k=1000`
      );

      if (!sceneResponse.ok) {
        throw new Error("Failed to fetch scene frames");
      }

      const sceneData = await sceneResponse.json();
      const frameIds = sceneData.data.map((frame: AVDataRecord) => frame.id);

      // Update all frames in the scene
      const updatePromises = frameIds.map((id: number) =>
        fetch(`${apiUrl}/api/v1/data/${id}`, {
          method: "PUT",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(updateData),
        })
      );

      await Promise.all(updatePromises);

      // Refresh current record
      const updatedResponse = await fetch(`${apiUrl}/api/v1/data/${currentRecord.id}`);
      if (updatedResponse.ok) {
        const updatedData = await updatedResponse.json();
        setCurrentRecord(updatedData);
        setMetadataWeather(updatedData.updated_metadata?.weather || updatedData.metadata?.weather || "");
        setMetadataSceneType(updatedData.updated_metadata?.scene_type || updatedData.metadata?.scene_type || "");
        setMetadataDaytime(updatedData.updated_metadata?.daytime || updatedData.metadata?.daytime || "");
      }

      setEditingMetadata(false);
      alert(`Successfully updated ${frameIds.length} frames in scene ${currentRecord.scene_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to apply metadata to scene");
    } finally {
      setSavingMetadata(false);
    }
  };

  /**
   * Save edited spatial relations
   */
  const handleSaveSpatialRelations = async (): Promise<void> => {
    if (!currentRecord || relationEdits.size === 0) return;

    try {
      // Log time spent before saving
      const timeSpent = (Date.now() - timerStartTimeRef.current) / 1000;
      if (timeSpent > 0.5) {
        await logTimeSpent(currentRecord.id, checkMode, timeSpent);
        timerStartTimeRef.current = Date.now(); // Reset timer
      }

      setSavingRelations(true);
      setError("");

      // Convert Map to API format
      const updates = Array.from(relationEdits.entries()).map(([key, relations]) => {
        const [source_track_id, target_track_id] = key.split("_");
        return { source_track_id, target_track_id, relations };
      });

      const response = await fetch(
        `${apiUrl}/api/v1/data/${currentRecord.id}/spatial-relations`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ updates, verify_by: username }),
        }
      );

      if (!response.ok) {
        throw new Error(`Failed to update spatial relations: ${response.statusText}`);
      }

      const result = await response.json();
      console.log("Spatial relations updated:", result);

      // Update the current record in place
      if (currentRecord.spatial_relations) {
        const updateMap = new Map(updates.map(u => [`${u.source_track_id}_${u.target_track_id}`, u.relations]));
        
        const updatedRelations = currentRecord.spatial_relations.map(rel => {
          const key = `${rel.source_object.track_id}_${rel.target_object.track_id}`;
          if (updateMap.has(key)) {
            return { ...rel, relations: updateMap.get(key)! };
          }
          return rel;
        });

        const updatedRecord = {
          ...currentRecord,
          spatial_relations: updatedRelations,
        };

        setCurrentRecord(updatedRecord);

        // Update cache
        setDataCache(prev => {
          const newCache = new Map(prev);
          newCache.set(currentOffset, updatedRecord);
          return newCache;
        });
      }

      // Clear edits and close editor
      setRelationEdits(new Map());
      setEditingRelations(false);
    } catch (err) {
      console.error("Error saving spatial relations:", err);
      setError(err instanceof Error ? err.message : "Failed to save spatial relations");
    } finally {
      setSavingRelations(false);
    }
  };

  /**
   * Save edited external interactions (from right panel editor)
   */
  const handleSaveExternalInteractions = async (): Promise<void> => {
    if (!currentRecord || externalInteractionEdits.size === 0) return;

    try {
      // Log time spent before saving
      const timeSpent = (Date.now() - timerStartTimeRef.current) / 1000;
      if (timeSpent > 0.5) {
        await logTimeSpent(currentRecord.id, checkMode, timeSpent);
        timerStartTimeRef.current = Date.now(); // Reset timer
      }

      setSavingRelations(true);
      setError("");

      // Apply edits to current external_interactions
      if (!currentRecord.external_interactions) {
        console.warn("No external_interactions in current record");
        return;
      }

      const updateMap = new Map(externalInteractionEdits);

      const updatedInteractions = currentRecord.external_interactions.map((interaction: any) => {
        const key = `${interaction.source_object.track_id}_${interaction.target_object.track_id}`;
        if (updateMap.has(key)) {
          return { ...interaction, interaction_type: updateMap.get(key)! };
        }
        return interaction;
      });

      // Send the FULL external_interactions list to backend (overwrite)
      const response = await fetch(
        `${apiUrl}/api/v1/data/${currentRecord.id}/external-interactions`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            external_interactions: updatedInteractions,
            verify_by: username
          }),
        }
      );

      if (!response.ok) {
        throw new Error(`Failed to update external interactions: ${response.statusText}`);
      }

      const result = await response.json();
      console.log("External interactions updated:", result);

      // Update local state
      const updatedRecord = {
        ...currentRecord,
        external_interactions: updatedInteractions,
        verified_external_interactions: 1,
        verify_by: username || currentRecord.verify_by || "",
      };

      setCurrentRecord(updatedRecord);

      // Update cache
      setDataCache(prev => {
        const newCache = new Map(prev);
        newCache.set(currentOffset, updatedRecord);
        return newCache;
      });

      // Clear edits and close editor
      setExternalInteractionEdits(new Map());
      setEditingExternalInteractions(false);
    } catch (err) {
      console.error("Error saving external interactions:", err);
      setError(err instanceof Error ? err.message : "Failed to save external interactions");
    } finally {
      setSavingRelations(false);
    }
  };

  /**
   * Handle relation updates from LidarViewer (direct click editing)
   */
  const handleRelationsUpdate = async (updates: Array<{ source_track_id: string; target_track_id: string; relations: string[] }>) => {
    if (!currentRecord) return;

    try {
      setSavingRelations(true);
      setError("");

      const response = await fetch(
        `${apiUrl}/api/v1/data/${currentRecord.id}/spatial-relations`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ updates, verify_by: username }),
        }
      );

      if (!response.ok) {
        throw new Error(`Failed to update spatial relations: ${response.statusText}`);
      }

      const result = await response.json();
      console.log("Spatial relations updated:", result);

      // Update the current record's spatial_relations in place without full refresh
      // This keeps the selected object and doesn't reset the view
      if (currentRecord.spatial_relations) {
        const updateMap = new Map(updates.map(u => [`${u.source_track_id}_${u.target_track_id}`, u.relations]));
        
        const updatedRelations = currentRecord.spatial_relations.map(rel => {
          const key = `${rel.source_object.track_id}_${rel.target_object.track_id}`;
          if (updateMap.has(key)) {
            return { ...rel, relations: updateMap.get(key)! };
          }
          return rel;
        });

        const updatedRecord = {
          ...currentRecord,
          spatial_relations: updatedRelations,
        };

        setCurrentRecord(updatedRecord);

        // Update cache with the updated record
        setDataCache(prev => {
          const newCache = new Map(prev);
          newCache.set(currentOffset, updatedRecord);
          return newCache;
        });
      }
    } catch (err) {
      console.error("Error saving spatial relations:", err);
      setError(err instanceof Error ? err.message : "Failed to save spatial relations");
    } finally {
      setSavingRelations(false);
    }
  };

  /**
   * Save & verify current frame (saves entire spatial_relations to backend)
   */
  const handleVerifyFrame = async (): Promise<void> => {
    if (!currentRecord || !currentRecord.spatial_relations) return;

    try {
      // Log time spent before saving
      const timeSpent = (Date.now() - timerStartTimeRef.current) / 1000;
      if (timeSpent > 0.5) {
        await logTimeSpent(currentRecord.id, checkMode, timeSpent);
        timerStartTimeRef.current = Date.now(); // Reset timer
      }

      setVerifyingFrame(true);
      setError("");

      // Send the entire current spatial_relations array (including all edits) to backend
      const response = await fetch(
        `${apiUrl}/api/v1/data/${currentRecord.id}/verify-spatial`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            spatial_relations: currentRecord.spatial_relations,
            verify_by: username,
          }),
        }
      );

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`Failed to save frame: ${response.status} ${response.statusText}`, errorText);
        throw new Error(`Failed to save frame: ${response.statusText}`);
      }

      const responseText = await response.text();
      console.log("Raw response:", responseText);
      
      let result;
      try {
        result = JSON.parse(responseText);
      } catch (jsonError) {
        console.error("Failed to parse JSON response:", jsonError);
        console.error("Response was:", responseText);
        throw new Error("Invalid response from server");
      }
      
      console.log("Frame saved and verified:", result);

      // Update current record to mark as verified
      const verifiedRecord = {
        ...currentRecord,
        verified_spatial_relations: 1,
      };
      setCurrentRecord(verifiedRecord);

      // Update cache with verified status
      setDataCache(prev => {
        const newCache = new Map(prev);
        newCache.set(currentOffset, verifiedRecord);
        return newCache;
      });

      // Move to next frame automatically after saving
      await handleNext();
    } catch (err) {
      console.error("Error saving frame:", err);
      setError(err instanceof Error ? err.message : "Failed to save frame");
    } finally {
      setVerifyingFrame(false);
    }
  };

  /**
   * Mark current frame as done for the current annotation mode
   */
  const handleMarkDone = async (): Promise<void> => {
    if (!currentRecord) return;

    try {
      // Log time spent before marking done
      const timeSpent = (Date.now() - timerStartTimeRef.current) / 1000;
      if (timeSpent > 0.5) {
        await logTimeSpent(currentRecord.id, checkMode, timeSpent);
        timerStartTimeRef.current = Date.now(); // Reset timer
      }

      setMarkingDone(true);
      setError("");

      // Determine which field to mark as verified based on current mode
      const verificationField = {
        metadata: "verified_metadata",
        spatial: "verified_spatial_relations",
        internal: "verified_internal_interactions",
        external: "verified_external_interactions",
        occlusion: "verified_metadata", // Map to metadata for now
        all: "verified_metadata", // Default to metadata for "all" mode
      }[checkMode] as string;

      const updateData = {
        [verificationField]: 1,
        verify_by: username,
      };

      console.log(`[Mark Done] Sending request for record ${currentRecord.id}, field: ${verificationField}`);

      const response = await fetch(
        `${apiUrl}/api/v1/data/${currentRecord.id}`,
        {
          method: "PUT",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(updateData),
        }
      );

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`[Mark Done] Error response:`, errorText);
        throw new Error(`Failed to mark as done: ${response.statusText} - ${errorText}`);
      }

      // Parse response JSON with error handling
      let responseData;
      try {
        const responseText = await response.text();
        console.log(`[Mark Done] Response text length: ${responseText.length}`);
        
        if (!responseText || responseText.trim().length === 0) {
          console.warn(`[Mark Done] Empty response body, but status was OK`);
          responseData = null;
        } else {
          responseData = JSON.parse(responseText);
          console.log(`[Mark Done] Successfully parsed response`);
        }
      } catch (parseErr) {
        console.error(`[Mark Done] JSON parse error:`, parseErr);
        console.error(`[Mark Done] Response was OK but failed to parse JSON`);
        // Don't throw - the update likely succeeded on the backend
        responseData = null;
      }

      // Update local record
      const updatedRecord = {
        ...currentRecord,
        [verificationField]: 1,
        verify_by: username,
      };
      setCurrentRecord(updatedRecord);

      // Update cache
      setDataCache(prev => {
        const newCache = new Map(prev);
        newCache.set(currentOffset, updatedRecord);
        return newCache;
      });

      console.log(`✅ Marked ${checkMode} as done for record ${currentRecord.id}`);
    } catch (err) {
      console.error("Error marking as done:", err);
      setError(err instanceof Error ? err.message : "Failed to mark as done");
    } finally {
      setMarkingDone(false);
    }
  };

  /**
   * Navigate to next uncompleted metadata scene
   */
  const handleNextUncompletedScene = async (): Promise<void> => {
    if (!currentRecord) return;

    try {
      setLoading(true);
      setError("");

      // Build query parameters
      const params = new URLSearchParams();
      params.append("current_scene_id", currentRecord.scene_id);
      
      // Apply source filter if any
      if (filterSource.length > 0) {
        filterSource.forEach(src => params.append("source", src));
      }

      const response = await fetch(
        `${apiUrl}/api/v1/scenes/next-uncompleted-metadata?${params.toString()}`
      );

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `Failed to find next uncompleted scene: ${response.statusText}`);
      }

      const result = await response.json();
      
      console.log("Next uncompleted metadata scene:", result);
      
      // Navigate to the target frame using direct ID
      await fetchData(result.target_id, 'direct', result.global_offset);
      
      console.log(`✅ Navigated to next uncompleted scene: ${result.scene_id}`);
    } catch (err) {
      console.error("Error navigating to next uncompleted scene:", err);
      setError(err instanceof Error ? err.message : "Failed to navigate to next uncompleted scene");
    } finally {
      setLoading(false);
    }
  };

  /**
   * Handle internal interactions update
   */
  const handleInternalInteractionsUpdate = async (updates: Array<{ track_id: string; actions: string[] }>): Promise<void> => {
    if (!currentRecord) return;

    try {
      // Log time spent before saving
      const timeSpent = (Date.now() - timerStartTimeRef.current) / 1000;
      if (timeSpent > 0.5) {
        await logTimeSpent(currentRecord.id, checkMode, timeSpent);
        timerStartTimeRef.current = Date.now(); // Reset timer
      }

      console.log("[Internal Interactions] Saving updates:", updates);
      setSavingRelations(true); // Reuse the same loading state
      setError("");

      // Apply updates to current internal_interactions to get the full list
      if (!currentRecord.internal_interactions) {
        console.warn("[Internal Interactions] No internal_interactions in current record");
        return;
      }

      const updateMap = new Map(updates.map(u => [u.track_id, u.actions]));
      
      const updatedInteractions = currentRecord.internal_interactions.map((interaction: any) => {
        if (updateMap.has(interaction.track_id)) {
          console.log(`[Internal Interactions] Updating track ${interaction.track_id} with actions:`, updateMap.get(interaction.track_id));
          return { ...interaction, actions: updateMap.get(interaction.track_id)! };
        }
        return interaction;
      });

      // Send the FULL internal_interactions list to backend (overwrite)
      const response = await fetch(
        `${apiUrl}/api/v1/data/${currentRecord.id}/internal-interactions`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ 
            internal_interactions: updatedInteractions,
            verify_by: username 
          }),
        }
      );

      if (!response.ok) {
        throw new Error(`Failed to update internal interactions: ${response.statusText}`);
      }

      const result = await response.json();
      console.log("[Internal Interactions] Backend response:", result);

      // Update local state
      const updatedRecord = {
        ...currentRecord,
        internal_interactions: updatedInteractions,
        verified_internal_interactions: 1,
        verify_by: username || currentRecord.verify_by || "",
      };

      console.log("[Internal Interactions] Updated record:", updatedRecord.internal_interactions);
      setCurrentRecord(updatedRecord);

      // Update cache
      setDataCache(prev => {
        const newCache = new Map(prev);
        newCache.set(currentOffset, updatedRecord);
        return newCache;
      });

      console.log("[Internal Interactions] State and cache updated successfully");
    } catch (err) {
      console.error("Error saving internal interactions:", err);
      setError(err instanceof Error ? err.message : "Failed to save internal interactions");
    } finally {
      setSavingRelations(false);
    }
  };

  /**
   * Handle external interactions update (individual edits)
   */
  const handleExternalInteractionsUpdate = async (updates: Array<{ source_track_id: string; target_track_id: string; interaction_type: string }>): Promise<void> => {
    if (!currentRecord) return;

    try {
      console.log("[External Interactions] Saving updates:", updates);
      setSavingRelations(true); // Reuse the same loading state
      setError("");

      // Apply updates to current external_interactions to get the full list
      if (!currentRecord.external_interactions) {
        console.warn("[External Interactions] No external_interactions in current record");
        return;
      }

      const updateMap = new Map(updates.map(u => [`${u.source_track_id}_${u.target_track_id}`, u.interaction_type]));
      
      const updatedInteractions = currentRecord.external_interactions
        .map((interaction: any) => {
          const key = `${interaction.source_object.track_id}_${interaction.target_object.track_id}`;
          if (updateMap.has(key)) {
            const newType = updateMap.get(key)!;
            console.log(`[External Interactions] Updating ${key} with interaction_type:`, newType);
            return { 
              ...interaction, 
              interaction_type: newType 
            };
          }
          return interaction;
        })
        .filter((interaction: any) => interaction.interaction_type !== ''); // Remove interactions with empty type

      // Send the FULL external_interactions list to backend (overwrite)
      const response = await fetch(
        `${apiUrl}/api/v1/data/${currentRecord.id}/external-interactions`,
        {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ 
            external_interactions: updatedInteractions,
            verify_by: username 
          }),
        }
      );

      if (!response.ok) {
        throw new Error(`Failed to update external interactions: ${response.statusText}`);
      }

      const result = await response.json();
      console.log("[External Interactions] Backend response:", result);

      // Update local state
      const updatedRecord = {
        ...currentRecord,
        external_interactions: updatedInteractions,
        verified_external_interactions: 1,
        verify_by: username || currentRecord.verify_by || "",
      };

      console.log("[External Interactions] Updated record:", updatedRecord.external_interactions);
      setCurrentRecord(updatedRecord);

      // Update cache
      setDataCache(prev => {
        const newCache = new Map(prev);
        newCache.set(currentOffset, updatedRecord);
        return newCache;
      });

      console.log("[External Interactions] State and cache updated successfully");
    } catch (err) {
      console.error("Error saving external interactions:", err);
      setError(err instanceof Error ? err.message : "Failed to save external interactions");
    } finally {
      setSavingRelations(false);
    }
  };

  /**
   * Navigate to previous scene - use direct ID lookup with offset tracking
   */
  const handlePreviousScene = async (): Promise<void> => {
    if (!currentRecord || !currentRecord.scene_id) return;

    try {
      setLoading(true);
      setError("");

      // Build query params with source filter
      const params = new URLSearchParams();
      params.append("current_scene_id", currentRecord.scene_id);
      params.append("direction", "previous");
      
      // Add source filter if active
      if (filterSource.length > 0) {
        filterSource.forEach((src: string) => params.append("source", src));
      }

      // Get both the target ID (for fast query) and global offset (for position tracking)
      const response = await fetch(
        `${apiUrl}/api/v1/scenes/jump?${params.toString()}`
      );

      if (!response.ok) {
        if (response.status === 404) {
          setError("No previous scene available");
          return;
        }
        throw new Error("Failed to navigate to previous scene");
      }

      const result = await response.json();
      console.log(`[Scene Navigation] Jumping to previous scene ${result.target_scene_id} at ID ${result.target_id}, offset ${result.global_offset}`);
      
      // Use direct ID lookup (instant) while maintaining global offset (position tracking)
      await fetchData(result.target_id, 'direct', result.global_offset);
    } catch (err) {
      console.error("Scene navigation error:", err);
      setError(err instanceof Error ? err.message : "Failed to navigate to previous scene");
    } finally {
      setLoading(false);
    }
  };

  /**
   * Navigate to next scene - use direct ID lookup with offset tracking
   */
  const handleNextScene = async (): Promise<void> => {
    if (!currentRecord || !currentRecord.scene_id) return;

    try {
      setLoading(true);
      setError("");
      
      console.log(`[Scene Navigation] Current scene: ${currentRecord.scene_id}`);

      // Build query params with source filter
      const params = new URLSearchParams();
      params.append("current_scene_id", currentRecord.scene_id);
      params.append("direction", "next");
      
      // Add source filter if active
      if (filterSource.length > 0) {
        filterSource.forEach((src: string) => params.append("source", src));
      }

      // Get both the target ID (for fast query) and global offset (for position tracking)
      const response = await fetch(
        `${apiUrl}/api/v1/scenes/jump?${params.toString()}`
      );

      if (!response.ok) {
        if (response.status === 404) {
          setError("No next scene available");
          return;
        }
        throw new Error("Failed to navigate to next scene");
      }

      const result = await response.json();
      console.log(`[Scene Navigation] Jumped to scene ${result.target_scene_id} at ID ${result.target_id}, offset ${result.global_offset}`);
      
      // Use direct ID lookup (instant) while maintaining global offset (position tracking)
      await fetchData(result.target_id, 'direct', result.global_offset);
    } catch (err) {
      console.error("Scene navigation error:", err);
      setError(err instanceof Error ? err.message : "Failed to navigate to next scene");
    } finally {
    setLoading(false);
    }
  };

  /**
   * Navigate to next track ID
   */
  const handleNextTrackId = () => {
    if (availableTrackIds.length === 0) return;
    
    const currentIndex = availableTrackIds.indexOf(selectedTrackId);
    if (currentIndex === -1 || selectedTrackId === "all") {
      // Start from first track ID
      setSelectedTrackId(availableTrackIds[0]);
    } else {
      // Move to next track ID (wrap around)
      const nextIndex = (currentIndex + 1) % availableTrackIds.length;
      setSelectedTrackId(availableTrackIds[nextIndex]);
    }
  };

  /**
   * Navigate to previous track ID
   */
  const handlePreviousTrackId = () => {
    if (availableTrackIds.length === 0) return;
    
    const currentIndex = availableTrackIds.indexOf(selectedTrackId);
    if (currentIndex === -1 || selectedTrackId === "all") {
      // Start from last track ID
      setSelectedTrackId(availableTrackIds[availableTrackIds.length - 1]);
    } else {
      // Move to previous track ID (wrap around)
      const prevIndex = (currentIndex - 1 + availableTrackIds.length) % availableTrackIds.length;
      setSelectedTrackId(availableTrackIds[prevIndex]);
    }
  };

  // Show loading while checking authentication
  if (authChecking || !isAuthenticated) {
    return (
      <div className="h-screen w-screen bg-gradient-to-br from-zinc-50 to-zinc-100 dark:from-zinc-900 dark:to-black font-sans flex items-center justify-center">
        <div className="text-xl text-zinc-600 dark:text-zinc-400">Loading...</div>
      </div>
    );
  }

  return (
    <div className="h-screen w-screen bg-gradient-to-br from-zinc-50 to-zinc-100 dark:from-zinc-900 dark:to-black font-sans overflow-hidden flex">
      {/* Left Side - Visualizations */}
      <div className="flex-1 flex flex-col p-4 gap-4 overflow-hidden">
        {/* LiDAR Visualization - Top (35% height) */}
        {currentRecord && !loading && showLidar && currentRecord.lidar?.path && (
          <div className="h-[35vh] bg-white dark:bg-zinc-800 rounded-xl shadow-lg overflow-hidden">
            <LidarViewer
              lidarPath={currentRecord.lidar.path}
              recordId={currentRecord.id}
              source={currentRecord.source}
              apiUrl={apiUrl}
              annotations={
                currentRecord.annotation?.bbox_3d
                  ? {
                      bbox_3d: currentRecord.annotation.bbox_3d,
                      label_3d: currentRecord.annotation.label_3d || [],
                      track_3d: currentRecord.annotation.track_3d || [],
                    }
                  : undefined
              }
              calibration={currentRecord.calibration}
              spatialRelations={currentRecord.spatial_relations || []}
              internalInteractions={currentRecord.internal_interactions || []}
              externalInteractions={currentRecord.external_interactions || []}
              occlusions={currentRecord.occlusions || []}
              showRelationships={checkMode === "spatial" || checkMode === "all"}
              showInternalInteractions={checkMode === "internal" || checkMode === "all"}
              showExternalInteractions={checkMode === "external" || checkMode === "all"}
              showOcclusions={checkMode === "occlusion" || checkMode === "all"}
              selectedTrackId={selectedTrackId}
              onRelationsUpdate={handleRelationsUpdate}
              onInternalInteractionsUpdate={handleInternalInteractionsUpdate}
              onExternalInteractionsUpdate={handleExternalInteractionsUpdate}
            />
          </div>
        )}

        {/* Camera Views - Bottom (remaining height) */}
        {currentRecord && !loading && (
          <div className="flex-1 bg-white dark:bg-zinc-800 rounded-xl shadow-lg p-4 overflow-hidden">
            {/* Dynamic grid based on number of cameras */}
            <div 
              className={`h-full ${
                currentCameraConfig.displayOrder.length === 4
                  ? "grid grid-cols-2 grid-rows-2 gap-2"  // 4 cameras: 2x2 (TruckScenes)
                  : currentCameraConfig.displayOrder.length === 5
                  ? "flex flex-col gap-2"  // 5 cameras: custom layout (Waymo)
                  : currentCameraConfig.displayOrder.length === 6
                  ? "grid grid-cols-3 grid-rows-2 gap-2"  // 6 cameras: 3x2 (nuScenes)
                  : currentCameraConfig.displayOrder.length === 7
                  ? "grid grid-cols-3 grid-rows-2 gap-2"  // 7 cameras: partial 3x2 + 1 (Argoverse2)
                  : currentCameraConfig.displayOrder.length <= 3
                  ? "grid grid-cols-3 grid-rows-1 gap-2"  // 1-3 cameras: single row
                  : "grid grid-cols-3 gap-2"  // Other: auto-flow in 3 columns
              }`}
            >
              {/* Special handling for Waymo (5 cameras) */}
              {currentCameraConfig.displayOrder.length === 5 ? (
                <>
                  {/* Top row: 3 cameras (FRONT_LEFT, FRONT, FRONT_RIGHT) */}
                  <div className="h-1/2 grid grid-cols-3 gap-2">
                    {currentCameraConfig.displayOrder.slice(0, 3).map((cameraName) => {
                      const imagePath = currentRecord.views[cameraName];
                      if (!imagePath) return null;

                      const apiIdx = getCameraApiIndex(cameraName);
                      const bboxes2d = apiIdx >= 0 ? (currentRecord.annotation?.bbox_2d?.[apiIdx] || []) : [];
                      const labels2d = apiIdx >= 0 ? (currentRecord.annotation?.label_2d?.[apiIdx] || []) : [];
                      const tracks2d = apiIdx >= 0 ? (currentRecord.annotation?.track_2d?.[apiIdx] || []) : [];

                      return (
                        <div
                          key={cameraName}
                          className="bg-zinc-50 dark:bg-zinc-900 rounded-lg overflow-hidden border border-zinc-200 dark:border-zinc-700 flex flex-col"
                        >
                          <div className="flex-1 relative bg-zinc-200 dark:bg-zinc-900 min-h-0">
                            <CameraWithBbox
                              imagePath={imagePath}
                              cameraName={cameraName.replace(/_/g, " ")}
                              bboxes={bboxes2d}
                              labels={labels2d}
                              trackIds={tracks2d}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  
                  {/* Bottom row: 2 cameras (SIDE_LEFT, SIDE_RIGHT) centered */}
                  <div className="h-1/2 grid grid-cols-3 gap-2">
                    <div></div> {/* Empty spacer */}
                    {currentCameraConfig.displayOrder.slice(3, 5).map((cameraName) => {
                      const imagePath = currentRecord.views[cameraName];
                      if (!imagePath) return null;

                      const apiIdx = getCameraApiIndex(cameraName);
                      const bboxes2d = apiIdx >= 0 ? (currentRecord.annotation?.bbox_2d?.[apiIdx] || []) : [];
                      const labels2d = apiIdx >= 0 ? (currentRecord.annotation?.label_2d?.[apiIdx] || []) : [];
                      const tracks2d = apiIdx >= 0 ? (currentRecord.annotation?.track_2d?.[apiIdx] || []) : [];

                      return (
                        <div
                          key={cameraName}
                          className="bg-zinc-50 dark:bg-zinc-900 rounded-lg overflow-hidden border border-zinc-200 dark:border-zinc-700 flex flex-col"
                        >
                          <div className="flex-1 relative bg-zinc-200 dark:bg-zinc-900 min-h-0">
                            <CameraWithBbox
                              imagePath={imagePath}
                              cameraName={cameraName.replace(/_/g, " ")}
                              bboxes={bboxes2d}
                              labels={labels2d}
                              trackIds={tracks2d}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </>
              ) : (
                /* Standard grid layout for other sources */
                currentCameraConfig.displayOrder.map((cameraName) => {
                const imagePath = currentRecord.views[cameraName];
                if (!imagePath) return null;

                // Get 2D bounding boxes for this camera using API order mapping
                const apiIdx = getCameraApiIndex(cameraName);
                const bboxes2d = apiIdx >= 0 ? (currentRecord.annotation?.bbox_2d?.[apiIdx] || []) : [];
                const labels2d = apiIdx >= 0 ? (currentRecord.annotation?.label_2d?.[apiIdx] || []) : [];
                const tracks2d = apiIdx >= 0 ? (currentRecord.annotation?.track_2d?.[apiIdx] || []) : [];

                return (
                  <div
                    key={cameraName}
                    className="bg-zinc-50 dark:bg-zinc-900 rounded-lg overflow-hidden border border-zinc-200 dark:border-zinc-700 flex flex-col"
                  >
                    <div className="flex-1 relative bg-zinc-200 dark:bg-zinc-900 min-h-0">
                      <CameraWithBbox
                        imagePath={imagePath}
                        cameraName={cameraName.replace(/_/g, " ")}
                        bboxes={bboxes2d}
                        labels={labels2d}
                        trackIds={tracks2d}
                      />
                    </div>
                  </div>
                );
                })
              )}
            </div>
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-4 border-zinc-300 border-t-blue-600 mx-auto mb-4"></div>
              <p className="text-zinc-600 dark:text-zinc-400">Loading data...</p>
            </div>
          </div>
        )}
      </div>

      {/* Right Side - Controls Panel */}
      <div className="w-96 bg-white dark:bg-zinc-800 shadow-xl flex flex-col overflow-y-auto">
        {/* Header */}
        <div className="p-4 border-b border-zinc-200 dark:border-zinc-700">
          <div className="flex items-start justify-between mb-2">
            <div>
              <h1 className="text-2xl font-bold text-zinc-900 dark:text-zinc-50 mb-1">
                DriveSpatial
              </h1>
              <p className="text-xs text-zinc-600 dark:text-zinc-400">
                AV Data Viewer
              </p>
            </div>
            <button
              onClick={handleLogout}
              className="px-3 py-1.5 bg-zinc-200 hover:bg-zinc-300 dark:bg-zinc-700 dark:hover:bg-zinc-600 text-zinc-900 dark:text-zinc-100 text-xs font-medium rounded transition-colors"
              title="Logout"
            >
              Logout
            </button>
          </div>
          <div className="mt-2 pt-2 border-t border-zinc-200 dark:border-zinc-700">
            <p className="text-xs text-zinc-600 dark:text-zinc-400">
              Logged in as: <span className="font-semibold text-zinc-900 dark:text-zinc-50">{username}</span>
            </p>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="m-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-3">
            <div className="flex items-start">
              <svg
                className="w-4 h-4 text-red-600 dark:text-red-400 mt-0.5 mr-2"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path
                  fillRule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                  clipRule="evenodd"
                />
              </svg>
              <div>
                <h3 className="text-xs font-medium text-red-800 dark:text-red-300">
                  Error
                </h3>
                <p className="mt-1 text-xs text-red-700 dark:text-red-400">
                  {error}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Record Information */}
        {currentRecord && (
          <div className="p-4 border-b border-zinc-200 dark:border-zinc-700">
            <div className="text-xs text-zinc-600 dark:text-zinc-400 space-y-2">
              <div className="flex justify-between items-center mb-3">
                <span className="font-semibold text-sm text-zinc-900 dark:text-zinc-50">
                  Record: {currentOffset + 1} / {totalRecords}
                </span>
                <span className="text-xs text-zinc-500 dark:text-zinc-400 font-mono">
                  ID: {currentRecord?.id || 'N/A'}
                </span>
              </div>
              <div>
                <span className="font-medium text-zinc-900 dark:text-zinc-50">Scene:</span>
                <div className="text-xs font-mono mt-1 break-all">{currentRecord.scene_id}</div>
              </div>
              <div>
                <span className="font-medium text-zinc-900 dark:text-zinc-50">Frame:</span>
                <div className="text-xs font-mono mt-1 break-all">{currentRecord.frame_id}</div>
              </div>
              <div>
                <span className="font-medium text-zinc-900 dark:text-zinc-50">Source:</span>{" "}
                {currentRecord.source}
              </div>
              
          {/* Timer Display - Isolated Component */}
          {/* <Timer 
            checkMode={checkMode}
            recordId={currentRecord?.id ?? null}
            timerStartTimeRef={timerStartTimeRef}
            onTimeLog={logTimeSpent}
          /> */}

              {/* Verification Status */}
              {/* <div className="mt-3 pt-3 border-t border-zinc-200 dark:border-zinc-700">
                <span className="font-medium text-zinc-900 dark:text-zinc-50 block mb-2">Verification Status:</span>
                <div className="flex flex-wrap gap-2">
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    currentRecord.verified_metadata === 1
                      ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                      : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
                  }`}>
                    Metadata: {currentRecord.verified_metadata === 1 ? "✓" : "✗"}
                  </span>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    currentRecord.verified_spatial_relations === 1
                      ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                      : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
                  }`}>
                    Spatial: {currentRecord.verified_spatial_relations === 1 ? "✓" : "✗"}
                  </span>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    currentRecord.verified_internal_interactions === 1
                      ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                      : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
                  }`}>
                    Actions: {currentRecord.verified_internal_interactions === 1 ? "✓" : "✗"}
                  </span>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    currentRecord.verified_external_interactions === 1
                      ? "bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                      : "bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200"
                  }`}>
                    External: {currentRecord.verified_external_interactions === 1 ? "✓" : "✗"}
                  </span>
                </div>
              </div> */}
            </div>
          </div>
        )}
         {/* Data Source Filter - Button Style */}
         <div className="p-4 border-b border-zinc-200 dark:border-zinc-700">
          <label className="block text-sm font-semibold text-zinc-900 dark:text-zinc-50 mb-3">
            Filter by Data Source
          </label>
          
          {/* Source Buttons */}
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => handleSourceToggle("nuScenes")}
              className={`px-3 py-2 rounded-lg text-xs font-semibold transition-all ${
                filterSource.includes("nuScenes")
                  ? "bg-emerald-500 text-white shadow-lg ring-2 ring-emerald-300"
                  : "bg-zinc-100 dark:bg-zinc-700 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-600"
              }`}
            >
              nuScenes
            </button>
            <button
              onClick={() => handleSourceToggle("waymo")}
              className={`px-3 py-2 rounded-lg text-xs font-semibold transition-all ${
                filterSource.includes("waymo")
                  ? "bg-sky-500 text-white shadow-lg ring-2 ring-sky-300"
                  : "bg-zinc-100 dark:bg-zinc-700 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-600"
              }`}
            >
              Waymo
            </button>
            <button
              onClick={() => handleSourceToggle("truckscenes")}
              className={`px-3 py-2 rounded-lg text-xs font-semibold transition-all ${
                filterSource.includes("truckscenes")
                  ? "bg-violet-500 text-white shadow-lg ring-2 ring-violet-300"
                  : "bg-zinc-100 dark:bg-zinc-700 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-600"
              }`}
            >
              TruckScenes
            </button>
            <button
              onClick={() => handleSourceToggle("once")}
              className={`px-3 py-2 rounded-lg text-xs font-semibold transition-all ${
                filterSource.includes("once")
                  ? "bg-orange-500 text-white shadow-lg ring-2 ring-orange-300"
                  : "bg-zinc-100 dark:bg-zinc-700 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-600"
              }`}
            >
              ONCE
            </button>
            <button
              onClick={() => handleSourceToggle("argoverse2")}
              className={`px-3 py-2 rounded-lg text-xs font-semibold transition-all ${
                filterSource.includes("argoverse2")
                  ? "bg-pink-500 text-white shadow-lg ring-2 ring-pink-300"
                  : "bg-zinc-100 dark:bg-zinc-700 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-600"
              }`}
            >
              Argoverse2
            </button>
          </div>

          {/* Selected count and clear */}
          {filterSource.length > 0 && (
            <div className="mt-3 flex items-center justify-between px-3 py-2 bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-lg">
              <span className="text-xs text-zinc-600 dark:text-zinc-400">
                {filterSource.length} source{filterSource.length !== 1 ? 's' : ''} selected
              </span>
              <button
                onClick={() => setFilterSource([])}
                className="text-xs text-blue-600 dark:text-blue-400 hover:underline font-medium"
              >
                Clear all
              </button>
            </div>
          )}
        </div>

        {/* Scene ID Filter */}
        <div className="p-4 border-b border-zinc-200 dark:border-zinc-700">
          <label className="block text-sm font-semibold text-zinc-900 dark:text-zinc-50 mb-2">
            Filter by Scene ID
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              value={filterSceneId}
              onChange={(e) => setFilterSceneId(e.target.value)}
              placeholder="Enter scene ID..."
              className="flex-1 px-3 py-2 text-sm bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-700 rounded-lg text-zinc-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            {filterSceneId && (
              <button
                onClick={() => setFilterSceneId("")}
                className="px-3 py-2 bg-zinc-200 hover:bg-zinc-300 dark:bg-zinc-700 dark:hover:bg-zinc-600 text-zinc-900 dark:text-zinc-100 text-sm font-medium rounded-lg transition-colors"
                title="Clear filter"
              >
                ✕
              </button>
            )}
          </div>
          {filterSceneId && (
            <p className="mt-2 text-xs text-zinc-500 dark:text-zinc-400">
              Filtering by: <span className="font-mono font-semibold">{filterSceneId}</span>
            </p>
          )}
        </div>

        {/* Metadata Information - Show in metadata/all mode OR when filtering by scene */}
        {currentRecord && currentRecord.metadata && (checkMode === "metadata" || checkMode === "all" || filterSceneId.trim()) && (
          <div className="p-4 border-b border-zinc-200 dark:border-zinc-700">
            <div className="flex justify-between items-center mb-3">
              <h3 className="text-sm font-semibold text-zinc-900 dark:text-zinc-50">
                Scene Metadata
              </h3>
              {!editingMetadata && (
                <button
                  onClick={() => setEditingMetadata(true)}
                  className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white text-xs font-medium rounded transition-colors"
                >
                  Edit
                </button>
              )}
            </div>
            <div className="space-y-3">
              {/* First Row: Daytime and Weather */}
              <div className="grid grid-cols-2 gap-3">
                {/* Daytime - Editable */}
                <div>
                  <span className="block text-xs font-medium text-zinc-600 dark:text-zinc-500 mb-1">
                    Daytime
                  </span>
                  {editingMetadata ? (
                    <select
                      value={metadataDaytime}
                      onChange={(e) => setMetadataDaytime(e.target.value)}
                      className="w-full px-2 py-1 text-xs bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-700 rounded text-zinc-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="">Select...</option>
                      <option value="morning">Morning</option>
                      <option value="noon">Noon</option>
                      <option value="afternoon">Afternoon</option>
                      <option value="night">Night</option>
                    </select>
                  ) : (
                    <div className="px-2 py-1.5 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded">
                      <span className="text-sm font-semibold text-blue-900 dark:text-blue-200 capitalize">
                        {(currentRecord.updated_metadata?.daytime || currentRecord.metadata?.daytime) || "Not set"}
                      </span>
                    </div>
                  )}
                </div>

                {/* Weather - Editable */}
                <div>
                  <span className="block text-xs font-medium text-zinc-600 dark:text-zinc-500 mb-1">
                    Weather
                  </span>
                  {editingMetadata ? (
                    <select
                      value={metadataWeather}
                      onChange={(e) => setMetadataWeather(e.target.value)}
                      className="w-full px-2 py-1 text-xs bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-700 rounded text-zinc-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="">Select...</option>
                      <option value="clear">Clear</option>
                      <option value="sunny">Sunny</option>
                      <option value="cloudy">Cloudy</option>
                      <option value="overcast">Overcast</option>
                      <option value="rain">Rain</option>
                      <option value="snow">Snow</option>
                      <option value="fog">Fog</option>
                      <option value="hail">Hail</option>
                      <option value="other">Other</option>
                    </select>
                  ) : (
                    <div className="px-2 py-1.5 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded">
                      <span className="text-sm font-semibold text-green-900 dark:text-green-200 capitalize">
                        {(currentRecord.updated_metadata?.weather || currentRecord.metadata?.weather) || "Not set"}
                      </span>
                    </div>
                  )}
                </div>
              </div>

              {/* Second Row: Scene Type */}
              <div>
                <span className="block text-xs font-medium text-zinc-600 dark:text-zinc-500 mb-1">
                  Scene Type
                </span>
                {editingMetadata ? (
                  <select
                    value={metadataSceneType}
                    onChange={(e) => setMetadataSceneType(e.target.value)}
                    className="w-full px-2 py-1 text-xs bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-700 rounded text-zinc-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">Select...</option>
                    <option value="cross_intersection">Cross Intersection</option>
                    <option value="t_intersection">T-Intersection</option>
                    <option value="y_intersection">Y-Intersection</option>
                    <option value="skewed_intersection">Skewed Intersection</option>
                    <option value="multi_leg_intersection">Multi-Leg Intersection</option>
                    <option value="straight_road">Straight Road</option>
                    <option value="s_curve_road">S-Curve Road</option>
                  </select>
                ) : (
                  <div className="px-2 py-1.5 bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded">
                    <span className="text-sm font-semibold text-purple-900 dark:text-purple-200 capitalize">
                      {((currentRecord.updated_metadata?.scene_type || currentRecord.metadata?.scene_type)?.replace(/_/g, " ")) || "Not set"}
                    </span>
                  </div>
                )}
              </div>

              {/* {currentRecord.metadata.location && (
                <div>
                  <span className="block text-xs font-medium text-zinc-600 dark:text-zinc-500 mb-1">
                    Location
                  </span>
                  <div className="px-2 py-1.5 bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded">
                    <span className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
                      {currentRecord.metadata.location}
                    </span>
                  </div>
                </div>
              )} */}
              {currentRecord.metadata.description && (
                <div>
                  <span className="block text-xs font-medium text-zinc-600 dark:text-zinc-500 mb-1">
                    Description
                  </span>
                  <div className="px-2 py-1.5 bg-zinc-50 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded">
                    <span className="text-sm font-medium text-zinc-900 dark:text-zinc-100">
                      {currentRecord.metadata.description}
                    </span>
                  </div>
                </div>
              )}

              {/* Quick Navigation - Show when NOT editing */}
              {/* {!editingMetadata && checkMode === "metadata" && (
                <div className="pt-2">
                  <button
                    onClick={handleNextUncompletedScene}
                    disabled={loading || !currentRecord}
                    className="w-full px-3 py-2 bg-gradient-to-r from-indigo-500 to-purple-500 hover:from-indigo-600 hover:to-purple-600 disabled:from-zinc-400 disabled:to-zinc-400 disabled:cursor-not-allowed text-white text-xs font-semibold rounded transition-all flex items-center justify-center gap-2 shadow-md"
                  >
                    {loading ? (
                      <>
                        <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                        Finding...
                      </>
                    ) : (
                      <>
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                        </svg>
                        Next Uncompleted Scene
                      </>
                    )}
                  </button>
                </div>
              )} */}

              {/* Save/Cancel buttons */}
              {editingMetadata && (
                <div className="flex gap-2 pt-2">
                  {/* <button
                    onClick={handleUpdateMetadata}
                    disabled={savingMetadata}
                    className="flex-1 px-3 py-2 bg-green-600 hover:bg-green-700 disabled:bg-zinc-400 disabled:cursor-not-allowed text-white text-xs font-medium rounded transition-colors flex items-center justify-center gap-2"
                  >
                    {savingMetadata && (
                      <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                    )}
                    {savingMetadata ? "Saving..." : "Save & Verify"}
                  </button> */}
                  <button
                    onClick={handleApplyToScene}
                    disabled={savingMetadata}
                    className="flex-1 px-3 py-2 bg-green-600 hover:bg-green-700  disabled:bg-zinc-400 disabled:cursor-not-allowed text-white text-xs font-medium rounded transition-colors flex items-center justify-center gap-2"
                  >
                    {savingMetadata && (
                      <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                    )}
                    {savingMetadata ? "Applying to Scene..." : "Apply to Scene"}
                  </button>
                  <button
                    onClick={() => {
                      setEditingMetadata(false);
                      setMetadataWeather(currentRecord.updated_metadata?.weather || currentRecord.metadata?.weather || "");
                      setMetadataSceneType(currentRecord.updated_metadata?.scene_type || currentRecord.metadata?.scene_type || "");
                      setMetadataDaytime(currentRecord.updated_metadata?.daytime || currentRecord.metadata?.daytime || "");
                    }}
                    disabled={savingMetadata}
                    className="px-3 py-2 bg-zinc-200 hover:bg-zinc-300 dark:bg-zinc-700 dark:hover:bg-zinc-600 text-zinc-900 dark:text-zinc-100 text-xs font-medium rounded transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              )}
            </div>
          </div>
        )}

        {/* API Configuration */}
        {/* <div className="p-4 border-b border-zinc-200 dark:border-zinc-700">
          <label className="block text-xs font-medium text-zinc-700 dark:text-zinc-300 mb-2">
            API URL
          </label>
          <div className="flex gap-2">
            <input
              type="text"
              value={apiUrl}
              onChange={(e) => setApiUrl(e.target.value)}
              placeholder="http://localhost:8000"
              className="flex-1 px-3 py-2 text-xs bg-zinc-50 dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-700 rounded-lg text-zinc-900 dark:text-zinc-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button
              onClick={() => fetchData(currentOffset, 'offset')}
              disabled={loading}
              className="px-3 py-2 bg-zinc-600 hover:bg-zinc-700 disabled:bg-zinc-300 disabled:cursor-not-allowed text-white text-xs font-medium rounded-lg transition-colors"
            >
              ↻
            </button>
          </div>
        </div> */}

        {/* View Controls */}
        {/* LiDAR is always shown - commented out toggle
        {currentRecord && (
          <div className="p-4 border-b border-zinc-200 dark:border-zinc-700">
            <label className="flex items-center gap-2 cursor-pointer mb-3">
              <input
                type="checkbox"
                checked={showLidar}
                onChange={(e) => setShowLidar(e.target.checked)}
                className="w-4 h-4"
              />
              <span className="text-sm font-medium text-zinc-900 dark:text-zinc-50">
                Show LiDAR
              </span>
            </label>
          </div>
        )}
        */}

        {/* Annotation Mode Selection - Clickable Buttons */}
        <div className="p-4 border-b border-zinc-200 dark:border-zinc-700">
          <label className="block text-sm font-semibold text-zinc-900 dark:text-zinc-50 mb-3">
            Annotation Mode
          </label>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => setCheckMode("metadata")}
              className={`px-3 py-2 rounded-lg text-xs font-semibold transition-all ${
                checkMode === "metadata"
                  ? "bg-blue-500 text-white shadow-lg ring-2 ring-blue-300"
                  : "bg-zinc-100 dark:bg-zinc-700 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-600"
              }`}
            >
              Metadata
            </button>
            <button
              onClick={() => setCheckMode("spatial")}
              className={`px-3 py-2 rounded-lg text-xs font-semibold transition-all ${
                checkMode === "spatial"
                  ? "bg-cyan-500 text-white shadow-lg ring-2 ring-cyan-300"
                  : "bg-zinc-100 dark:bg-zinc-700 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-600"
              }`}
            >
              Spatial
            </button>
            <button
              onClick={() => setCheckMode("internal")}
              className={`px-3 py-2 rounded-lg text-xs font-semibold transition-all ${
                checkMode === "internal"
                  ? "bg-purple-500 text-white shadow-lg ring-2 ring-purple-300"
                  : "bg-zinc-100 dark:bg-zinc-700 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-600"
              }`}
            >
              Actions
            </button>
            <button
              onClick={() => setCheckMode("external")}
              className={`px-3 py-2 rounded-lg text-xs font-semibold transition-all ${
                checkMode === "external"
                  ? "bg-green-500 text-white shadow-lg ring-2 ring-green-300"
                  : "bg-zinc-100 dark:bg-zinc-700 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-600"
              }`}
            >
              Interactions
            </button>
            {/* <button
              onClick={() => setCheckMode("occlusion")}
              className={`px-3 py-2 rounded-lg text-xs font-semibold transition-all ${
                checkMode === "occlusion"
                  ? "bg-orange-500 text-white shadow-lg ring-2 ring-orange-300"
                  : "bg-zinc-100 dark:bg-zinc-700 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-600"
              }`}
            >
              Occlusion
            </button> */}
            {/* <button
              onClick={() => setCheckMode("all")}
              className={`px-3 py-2 rounded-lg text-xs font-semibold transition-all ${
                checkMode === "all"
                  ? "bg-amber-500 text-white shadow-lg ring-2 ring-amber-300"
                  : "bg-zinc-100 dark:bg-zinc-700 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-200 dark:hover:bg-zinc-600"
              }`}
            >
              All
            </button> */}
          </div>
          
          {/* Mark Done Button */}
          <div className="mt-3">
            <button
              onClick={handleMarkDone}
              disabled={markingDone || loading || !currentRecord}
              className="w-full px-4 py-3 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white text-sm font-semibold rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 shadow-lg"
            >
              {markingDone ? (
                <>
                  <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Marking...
                </>
              ) : (
                <>
                   Mark as Done
                </>
              )}
            </button>
          </div>
          
        </div>

        {/* Target Object ID Selection - Show in spatial, external, or all mode */}
        {(checkMode === "spatial" || checkMode === "external" || checkMode === "all") && (
          <div className="p-4 border-b border-zinc-200 dark:border-zinc-700">
            <label className="block text-sm font-semibold text-zinc-900 dark:text-zinc-50 mb-2">
              Source Object ID
            </label>
            <div className="space-y-2">
              {/* Display current selection */}
              <div className="px-3 py-2 bg-zinc-100 dark:bg-zinc-800 border border-zinc-200 dark:border-zinc-700 rounded-lg">
                <div className="text-xs text-zinc-500 dark:text-zinc-400 mb-1">Selected:</div>
                <div className="text-sm font-mono text-zinc-900 dark:text-zinc-100 truncate">
                  {selectedTrackId === "all" ? "All Objects" : selectedTrackId}
                </div>
                {selectedTrackId !== "all" && (
                  <div className="text-xs text-zinc-500 dark:text-zinc-400 mt-1">
                    {availableTrackIds.indexOf(selectedTrackId) + 1} / {availableTrackIds.length}
                  </div>
                )}
              </div>
              
              {/* Navigation buttons */}
              <div className="flex gap-2">
                <button
                  onClick={handlePreviousTrackId}
                  disabled={loading || availableTrackIds.length === 0}
                  className="flex-1 px-3 py-2 bg-zinc-200 hover:bg-zinc-300 dark:bg-zinc-700 dark:hover:bg-zinc-600 text-zinc-900 dark:text-zinc-100 text-sm font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  title="Previous object ID"
                >
                  ← Prev
                </button>
                <button
                  onClick={() => setSelectedTrackId("all")}
                  disabled={loading || selectedTrackId === "all"}
                  className="px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  title="Show all objects"
                >
                  All
                </button>
                <button
                  onClick={handleNextTrackId}
                  disabled={loading || availableTrackIds.length === 0}
                  className="flex-1 px-3 py-2 bg-zinc-200 hover:bg-zinc-300 dark:bg-zinc-700 dark:hover:bg-zinc-600 text-zinc-900 dark:text-zinc-100 text-sm font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  title="Next object ID"
                >
                  Next →
                </button>
              </div>

              {/* Stats */}
              <div className="text-xs text-zinc-500 dark:text-zinc-400 text-center">
                {availableTrackIds.length} objects in scene
              </div>
            </div>
          </div>
        )}

        {/* Save Frame Button - Always show in spatial or all mode */}
        {/* {(checkMode === "spatial" || checkMode === "all") && currentRecord && (
          <div className="p-4 border-b border-zinc-200 dark:border-zinc-700">
            <button
              onClick={handleVerifyFrame}
              disabled={verifyingFrame || loading}
              className="w-full px-4 py-3 bg-green-500 hover:bg-green-600 text-white text-sm font-semibold rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {verifyingFrame ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                  Saving...
                </>
              ) : (
                <>
                  💾 Save & Next
                </>
              )}
            </button>
            <div className="text-xs text-zinc-500 dark:text-zinc-400 text-center mt-2">
              Save all spatial relations and mark as verified
            </div>
          </div>
        )} */}

        {/* Spatial Relations Editor - Only show when target object is selected */}
        {(checkMode === "spatial" || checkMode === "all") && selectedTrackId !== "all" && currentRecord && currentRecord.spatial_relations && (
          <div className="p-4 border-b border-zinc-200 dark:border-zinc-700">
            <div className="flex items-center justify-between mb-3">
              <label className="block text-sm font-semibold text-zinc-900 dark:text-zinc-50">
                Edit Spatial Relations
              </label>
              <button
                onClick={() => setEditingRelations(!editingRelations)}
                className="px-3 py-1 text-xs bg-blue-500 hover:bg-blue-600 text-white rounded transition-colors"
              >
                {editingRelations ? "Cancel" : "Edit"}
              </button>
            </div>

           
          </div>
        )}

        {/* External Interactions Editor - Only show when source object is selected */}
        {(checkMode === "external" || checkMode === "all") && selectedTrackId !== "all" && currentRecord && currentRecord.external_interactions && (
          <div className="p-4 border-b border-zinc-200 dark:border-zinc-700">
            <div className="flex items-center justify-between mb-3">
              <label className="block text-sm font-semibold text-zinc-900 dark:text-zinc-50">
                Edit External Interactions
              </label>
              <button
                onClick={() => setEditingExternalInteractions(!editingExternalInteractions)}
                className="px-3 py-1 text-xs bg-green-500 hover:bg-green-600 text-white rounded transition-colors"
              >
                {editingExternalInteractions ? "Cancel" : "Edit"}
              </button>
            </div>

            {editingExternalInteractions ? (
              <div className="space-y-3">
                {/* Show interactions where selected object is source */}
                {currentRecord.external_interactions
                  .filter((interaction: any) => interaction.source_object.track_id === selectedTrackId)
                  .map((interaction: any, idx: number) => {
                    const key = `${interaction.source_object.track_id}_${interaction.target_object.track_id}`;
                    const currentInteractionType = externalInteractionEdits.get(key) || interaction.interaction_type;
                    
                    const availableInteractions = [
                      "lead", "follow", "overtake", "passing", "co_moving",
                      "approaching", "crossing", "waiting", "yielding"
                    ];
                    
                    const externalInteractionColors: { [key: string]: string } = {
                      'lead': '#10B981',
                      'follow': '#3B82F6',
                      'overtake': '#F59E0B',
                      'passing': '#06B6D4',
                      'co_moving': '#8B5CF6',
                      'approaching': '#EF4444',
                      'crossing': '#EC4899',
                      'waiting': '#6366F1',
                      'yielding': '#F97316',
                    };

                    const selectInteraction = (interactionType: string) => {
                      // Single selection only
                      const newEdits = new Map(externalInteractionEdits);
                      newEdits.set(key, interactionType);
                      setExternalInteractionEdits(newEdits);
                    };

                    return (
                      <div key={idx} className="p-3 bg-zinc-100 dark:bg-zinc-800 rounded-lg space-y-2">
                        <div className="text-xs font-semibold text-zinc-700 dark:text-zinc-300">
                          To: {interaction.target_object.label} ({interaction.target_object.track_id.substring(0, 8)}...)
                        </div>
                        <div className="flex flex-wrap gap-1">
                          {availableInteractions.map(interactionType => {
                            const color = externalInteractionColors[interactionType] || '#6B7280';
                            const isSelected = currentInteractionType === interactionType;
                            
                            return (
                              <button
                                key={interactionType}
                                onClick={() => selectInteraction(interactionType)}
                                style={{
                                  backgroundColor: isSelected ? color : undefined,
                                  borderColor: isSelected ? color : undefined,
                                  borderWidth: isSelected ? '2px' : undefined,
                                }}
                                className={`px-2 py-1 text-xs rounded transition-all font-semibold ${
                                  isSelected
                                    ? "text-white shadow-lg"
                                    : "bg-zinc-200 dark:bg-zinc-700 text-zinc-700 dark:text-zinc-300 hover:bg-zinc-300 dark:hover:bg-zinc-600"
                                }`}
                              >
                                {interactionType.replace(/_/g, " ")}
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    );
                  })}

                {/* Save button */}
                <button
                  onClick={handleSaveExternalInteractions}
                  disabled={savingRelations || externalInteractionEdits.size === 0}
                  className="w-full px-4 py-2 bg-green-500 hover:bg-green-600 text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  {savingRelations && (
                    <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                  )}
                  {savingRelations ? "Saving..." : `Save Changes (${externalInteractionEdits.size})`}
                </button>
              </div>
            ) : (
              <div className="text-xs text-zinc-500 dark:text-zinc-400 text-center py-2">
                Click "Edit" to modify interactions
              </div>
            )}
          </div>
        )}

       

        {/* Navigation Controls */}
        {currentRecord && (
          <div className="p-4 space-y-3">
            {/* Frame Navigation - Compact Arrow Buttons */}
            <div>
              <label className="block text-xs font-medium text-zinc-700 dark:text-zinc-300 mb-2">
                Frame Navigation
              </label>
              <div className="flex gap-2">
                <button
                  onClick={handlePrevious}
                  disabled={loading || !currentRecord}
                  className="flex-1 px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-zinc-300 disabled:cursor-not-allowed text-white font-medium rounded transition-colors text-sm"
                  title="Previous Frame"
                >
                  ← Prev
                </button>
                <button
                  onClick={handleNext}
                  disabled={loading || !currentRecord}
                  className="flex-1 px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-zinc-300 disabled:cursor-not-allowed text-white font-medium rounded transition-colors text-sm"
                  title="Next Frame"
                >
                  Next →
                </button>
              </div>
            </div>

            {/* Scene Navigation */}
            <div>
              <label className="block text-xs font-medium text-zinc-700 dark:text-zinc-300 mb-2">
                Scene Navigation
              </label>
              <div className="flex gap-2">
                <button
                  onClick={handlePreviousScene}
                  disabled={loading}
                  className="flex-1 px-3 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-zinc-300 disabled:cursor-not-allowed text-white font-medium rounded transition-colors text-sm"
                  title="Previous Scene"
                >
                  ⟪ Scene
                </button>
                <button
                  onClick={handleNextScene}
                  disabled={loading}
                  className="flex-1 px-3 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-zinc-300 disabled:cursor-not-allowed text-white font-medium rounded transition-colors text-sm"
                  title="Next Scene"
                >
                  Scene ⟫
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Instructions */}
        {!currentRecord && !loading && !error && (
          <div className="p-4">
            <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-blue-900 dark:text-blue-300 mb-2">
              Getting Started
            </h3>
              <ul className="space-y-2 text-xs text-blue-800 dark:text-blue-400">
              <li className="flex items-start">
                <span className="mr-2">1.</span>
                  <span>Make sure the FastAPI server is running at {apiUrl}</span>
              </li>
              <li className="flex items-start">
                <span className="mr-2">2.</span>
                  <span>
                    Ensure HPC credentials are set in .env.local file
                  </span>
              </li>
              <li className="flex items-start">
                <span className="mr-2">3.</span>
                  <span>Use Previous/Next buttons to navigate</span>
              </li>
            </ul>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
