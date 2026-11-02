"use client";

import { useRef, useEffect, useState } from "react";

interface LidarViewerProps {
  lidarPath: string;
  recordId: number;
  source: string; // Data source: "nuscenes", "waymo", "truckscenes", etc.
  apiUrl?: string; // API URL for backend requests
  annotations?: {
    bbox_3d: number[][];
    label_3d: string[];
    track_3d: string[];
  };
  calibration?: {
    ego_pose?: number[][];
    [key: string]: any;
  };
  spatialRelations?: Array<{
    source_object: { track_id: string; label: string };
    target_object: { track_id: string; label: string };
    relations: string[];
    rel_coords_in_source_frame: { forward: number; right: number; up: number };
    source_forward_vec_xy: { fx: number; fy: number };
  }>;
  internalInteractions?: Array<{
    track_id: string;
    label: string;
    actions: string[];
    details?: {
      velocity_magnitude?: number;
      velocity_x?: number;
      velocity_y?: number;
      displacement?: number;
      heading_change?: number;
      cos_angle?: number;
      lateral_displacement?: number;
      adjusted_lateral?: number;
      acceleration?: number;
      [key: string]: any;
    };
  }>;
  externalInteractions?: Array<{
    source_object: { track_id: string; label: string };
    target_object: { track_id: string; label: string };
    interaction_type: string;
    details?: any;
  }>;
  occlusions?: Array<{
    source_object: { track_id: string; label: string };
    target_object: { track_id: string; label: string };
    visibility: number;
  }>;
  showRelationships?: boolean;
  showInternalInteractions?: boolean;
  showExternalInteractions?: boolean;
  showOcclusions?: boolean;
  selectedTrackId?: string; // "all" or specific track ID
  onRelationsUpdate?: (updates: Array<{ source_track_id: string; target_track_id: string; relations: string[] }>) => void;
  onInternalInteractionsUpdate?: (updates: Array<{ track_id: string; actions: string[] }>) => void;
  onExternalInteractionsUpdate?: (updates: Array<{ source_track_id: string; target_track_id: string; interaction_type: string }>) => void;
}

export default function LidarViewer({ 
  lidarPath, 
  recordId, 
  source,
  apiUrl = "http://localhost:8000", // Default API URL
  annotations, 
  calibration,
  spatialRelations, 
  internalInteractions,
  externalInteractions,
  occlusions,
  showRelationships = false, 
  showInternalInteractions = false,
  showExternalInteractions = false,
  showOcclusions = false,
  selectedTrackId = "all", 
  onRelationsUpdate,
  onInternalInteractionsUpdate,
  onExternalInteractionsUpdate
}: LidarViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState({ count: 0, minZ: 0, maxZ: 0 });
  const [showAnnotations, setShowAnnotations] = useState(true);
  
  // Relationship selection and editing
  const [selectedRelationship, setSelectedRelationship] = useState<{
    source_track_id: string;
    target_track_id: string;
    source_label: string;
    target_label: string;
    relations: string[];
    position: { x: number; y: number }; // Screen position for editor
  } | null>(null);
  
  const availableRelations = [
    "ahead_of", "behind", "right_of", "left_of",
    "ahead_right_of", "ahead_left_of", "rear_right_of", "rear_left_of",
    "above", "below"
  ];
  
  // Internal interaction selection and editing
  const [selectedInteraction, setSelectedInteraction] = useState<{
    track_id: string;
    label: string;
    actions: string[];
    position: { x: number; y: number }; // Screen position for editor
  } | null>(null);
  
  const availableActions = [
    "stopped", "moving_forward", "moving_backward", "u_turn",
    "turn_left", "turn_right", "lane_change_left", "lane_change_right",
    "accelerate", "decelerate"
  ];
  
  // External interaction selection and editing
  const [selectedExternalInteraction, setSelectedExternalInteraction] = useState<{
    source_track_id: string;
    target_track_id: string;
    source_label: string;
    target_label: string;
    interaction_type: string;
    position: { x: number; y: number }; // Screen position for editor
  } | null>(null);
  
  const availableExternalInteractions = [
    "lead", "follow", "overtake", "passing", "co_moving",
    "approaching", "crossing", "waiting", "yielding"
  ];
  
  // Color map for each external interaction type
  const externalInteractionColors: { [key: string]: string } = {
    'lead': '#10B981',           // Green
    'follow': '#3B82F6',         // Blue
    'overtake': '#F59E0B',       // Amber
    'passing': '#06B6D4',        // Cyan
    'co_moving': '#8B5CF6',      // Violet
    'approaching': '#EF4444',    // Red
    'crossing': '#EC4899',       // Pink
    'waiting': '#6366F1',        // Indigo
    'yielding': '#F97316',       // Orange
  };
  
  // Color map for each action type
  const actionColors: { [key: string]: string } = {
    'stopped': '#EF4444',              // Red
    'moving_forward': '#10B981',       // Green
    'moving_backward': '#F59E0B',      // Amber
    'u_turn': '#8B5CF6',               // Violet
    'turn_left': '#3B82F6',            // Blue
    'turn_right': '#06B6D4',           // Cyan
    'lane_change_left': '#6366F1',     // Indigo
    'lane_change_right': '#14B8A6',    // Teal
    'accelerate': '#84CC16',           // Lime
    'decelerate': '#F97316',           // Orange
  };

  // Store camera position to preserve it across updates
  const cameraStateRef = useRef<{
    position: { x: number; y: number; z: number };
    target: { x: number; y: number; z: number };
    recordId: number;
    selectedTrackId: string; // Track which object we positioned for
  } | null>(null);

  /**
   * Helper function to clamp popup position within viewport bounds
   * Ensures popups are fully visible with padding from edges
   */
  const clampPopupPosition = (
    x: number, 
    y: number, 
    popupWidth: number = 350, 
    popupHeight: number = 450
  ): { x: number; y: number } => {
    const padding = 16; // Minimum padding from viewport edges
    const maxX = window.innerWidth - popupWidth - padding;
    const maxY = window.innerHeight - popupHeight - padding;
    
    return {
      x: Math.max(padding, Math.min(x, maxX)),
      y: Math.max(padding, Math.min(y, maxY))
    };
  };

  useEffect(() => {
    if (!containerRef.current) return;

    let scene: any, camera: any, renderer: any, controls: any;
    let animationId: number;
    let annotationGroup: any;
    let THREE: any;
    let isCleanedUp = false; // Flag to prevent operations after cleanup

    const initAndLoadLidar = async () => {
      try {
        setLoading(true);
        setError(null);

        // Dynamically import Three.js
        THREE = await import("three");
        const { OrbitControls } = await import("three/examples/jsm/controls/OrbitControls.js");

        const container = containerRef.current;
        if (!container || isCleanedUp) return;

        // Clear container safely
        while (container.firstChild) {
          container.removeChild(container.firstChild);
        }

        const width = container.clientWidth;
        const height = container.clientHeight;

        // Scene
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x18181b);

        // Camera - positioned behind the ego vehicle
        // Ego vehicle is at origin, facing in -Z direction (forward)
        camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
        
        // Restore camera position if we have saved state for this record AND same selected object
        if (cameraStateRef.current && 
            cameraStateRef.current.recordId === recordId &&
            cameraStateRef.current.selectedTrackId === selectedTrackId) {
          const saved = cameraStateRef.current;
          camera.position.set(saved.position.x, saved.position.y, saved.position.z);
          camera.lookAt(saved.target.x, saved.target.y, saved.target.z);
          console.log("Restored camera position:", saved.position);
        } else {
          // Default position
          camera.position.set(15, 5, 0); // Behind vehicle (positive Z), elevated
          camera.lookAt(0, 0, 0);
        }

        // Renderer
        renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(width, height);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        // Controls
        controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        
        // Restore target if we have saved state for same record AND same selected object
        if (cameraStateRef.current && 
            cameraStateRef.current.recordId === recordId &&
            cameraStateRef.current.selectedTrackId === selectedTrackId) {
          const saved = cameraStateRef.current;
          controls.target.set(saved.target.x, saved.target.y, saved.target.z);
        } else {
          controls.target.set(0, 0, 0);
        }
        controls.update();

        // Save camera position on any change (includes selected object in state)
        controls.addEventListener('change', () => {
          cameraStateRef.current = {
            position: {
              x: camera.position.x,
              y: camera.position.y,
              z: camera.position.z
            },
            target: {
              x: controls.target.x,
              y: controls.target.y,
              z: controls.target.z
            },
            recordId: recordId,
            selectedTrackId: selectedTrackId // Track which object this view is for
          };
        });

        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);

        // Grid and axes
        const gridHelper = new THREE.GridHelper(100, 100, 0x444444, 0x222222);
        scene.add(gridHelper);

        const axesHelper = new THREE.AxesHelper(5);
        scene.add(axesHelper);

        // Create annotation group
        annotationGroup = new THREE.Group();
        scene.add(annotationGroup);

        // Animation loop
        const animate = () => {
          // Check if cleanup has occurred or resources are invalid
          if (isCleanedUp || !renderer || !scene || !camera || !controls) {
            if (animationId) cancelAnimationFrame(animationId);
            return;
          }
          
          animationId = requestAnimationFrame(animate);
          controls.update();
          
          try {
            renderer.render(scene, camera);
          } catch (err) {
            console.error('[LiDAR] Render error:', err);
            if (animationId) cancelAnimationFrame(animationId);
          }
        };
        animate();

        // Raycaster for click detection
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        raycaster.params.Line = { threshold: 0.5 }; // Make lines easier to click
        
        // Handle clicks on relationships and actions
        const handleClick = (event: MouseEvent) => {
          if (!container) return;
          
          const rect = container.getBoundingClientRect();
          mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
          mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
          
          raycaster.setFromCamera(mouse, camera);
          const intersects = raycaster.intersectObjects(annotationGroup.children, true);
          
          if (intersects.length > 0) {
            for (const intersect of intersects) {
              const userData = (intersect.object as any).userData;
              
              // Handle relationship clicks (only when filtering by specific object)
              if (userData && userData.type === 'relationship' && selectedTrackId !== "all") {
                // Get screen position for editor popup
                const screenPos = new THREE.Vector3();
                screenPos.copy(userData.controlPoint);
                screenPos.project(camera);
                
                const x = (screenPos.x * 0.5 + 0.5) * rect.width;
                const y = (-(screenPos.y * 0.5) + 0.5) * rect.height;
                
                setSelectedRelationship({
                  source_track_id: userData.source_track_id,
                  target_track_id: userData.target_track_id,
                  source_label: userData.source_label,
                  target_label: userData.target_label,
                  relations: [...userData.relations],
                  position: { x, y }
                });
                setSelectedInteraction(null); // Close interaction editor if open
                break;
              }
              
              // Handle action sprite clicks
              if (userData && userData.type === 'action') {
                // Get screen position for editor popup
                const worldPos = (intersect.object as any).position;
                const screenPos = worldPos.clone();
                screenPos.copy(worldPos);
                screenPos.project(camera);
                
                const x = (screenPos.x * 0.5 + 0.5) * rect.width;
                const y = (-(screenPos.y * 0.5) + 0.5) * rect.height;
                
                setSelectedInteraction({
                  track_id: userData.track_id,
                  label: userData.label,
                  actions: [...userData.actions],
                  position: { x, y }
                });
                setSelectedRelationship(null); // Close relationship editor if open
                setSelectedExternalInteraction(null); // Close external interaction editor if open
                break;
              }
              
              // Handle external interaction clicks (only when filtering by specific object)
              if (userData && userData.type === 'externalInteraction' && selectedTrackId !== "all") {
                console.log('[Click] External interaction clicked:', userData);
                
                // Get screen position for editor popup
                const screenPos = new THREE.Vector3();
                screenPos.copy(userData.controlPoint);
                screenPos.project(camera);
                
                const x = (screenPos.x * 0.5 + 0.5) * rect.width;
                const y = (-(screenPos.y * 0.5) + 0.5) * rect.height;
                
                const interactionType = Array.isArray(userData.interactions) && userData.interactions.length > 0 
                  ? userData.interactions[0] 
                  : (userData.interaction_type || "");
                
                console.log('[Click] Opening editor with interaction_type:', interactionType);
                
                setSelectedExternalInteraction({
                  source_track_id: userData.source_track_id,
                  target_track_id: userData.target_track_id,
                  source_label: userData.source_label,
                  target_label: userData.target_label,
                  interaction_type: interactionType,
                  position: { x, y }
                });
                setSelectedRelationship(null); // Close relationship editor if open
                setSelectedInteraction(null); // Close interaction editor if open
                break;
              }
            }
          }
        };
        
        renderer.domElement.addEventListener('click', handleClick);

        // Handle window resize
        const handleResize = () => {
          if (!container) return;
          const w = container.clientWidth;
          const h = container.clientHeight;
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
          renderer.setSize(w, h);
        };
        window.addEventListener("resize", handleResize);

        // Try to load LiDAR point cloud data (optional - annotations will still render if this fails)
        let pointCloudLoaded = false;
        try {
          let buffer: ArrayBuffer;
          let text: string;
          
          // Check if this is a feather file (AV2/Argoverse2)
          if (lidarPath.endsWith(".feather")) {
            console.log("[LiDAR] Detected feather format (AV2), requesting conversion from backend");
            
            // Call backend to convert feather to PCD
            const conversionResponse = await fetch(
              `${apiUrl}/api/v1/lidar/convert-feather?path=${encodeURIComponent(lidarPath)}`
            );
            
            if (!conversionResponse.ok) {
              throw new Error(`Failed to convert feather file: ${conversionResponse.statusText}`);
            }
            
            const conversionData = await conversionResponse.json();
            text = conversionData.pcd_content;
            console.log(`[LiDAR] Converted feather to PCD: ${conversionData.num_points} points`);
            
            // Convert text to buffer for consistent processing
            buffer = new TextEncoder().encode(text).buffer;
          } else {
            // Standard file loading for PCD and BIN files
            const response = await fetch(`/api/hpc-lidar?path=${encodeURIComponent(lidarPath)}`);
            if (!response.ok) throw new Error(`Failed to fetch lidar: ${response.statusText}`);

            buffer = await response.arrayBuffer();
            console.log(`[LiDAR] Loaded ${buffer.byteLength} bytes from ${lidarPath}`);
          }

        // Parse based on format
        let points: Float32Array;
        let colors: Float32Array;

        if (lidarPath.endsWith(".feather") || (lidarPath.endsWith(".pcd") && !lidarPath.endsWith(".pcd.bin"))) {
          // Feather (converted to PCD) or standard PCD format
          console.log("[LiDAR] Parsing PCD format");
          
          // For feather files, text is already set; for PCD, decode the buffer
          if (!lidarPath.endsWith(".feather")) {
            text = new TextDecoder().decode(buffer);
          }
          
          const lines = text.split('\n');
          
          // Parse PCD header
          let dataFormat = 'ascii';
          let pointCount = 0;
          let dataStart = 0;
          let fields: string[] = [];
          let sizes: number[] = [];
          
          for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            if (line.startsWith('FIELDS')) {
              fields = line.split(/\s+/).slice(1);
              console.log(`[LiDAR] Fields: ${fields.join(', ')}`);
            }
            if (line.startsWith('SIZE')) {
              sizes = line.split(/\s+/).slice(1).map(s => parseInt(s));
            }
            if (line.startsWith('POINTS')) {
              pointCount = parseInt(line.split(/\s+/)[1]);
              console.log(`[LiDAR] Point count: ${pointCount}`);
            }
            if (line.startsWith('DATA')) {
              dataFormat = line.split(/\s+/)[1].toLowerCase();
              dataStart = i + 1;
              console.log(`[LiDAR] Data format: ${dataFormat}, starts at line ${dataStart}`);
              break;
            }
          }
          
          if (pointCount === 0) {
            throw new Error("No points found in PCD file");
          }
          
          points = new Float32Array(pointCount * 3);
          colors = new Float32Array(pointCount * 3);
          
          if (dataFormat === 'ascii') {
            // Parse ASCII PCD data
            let pointIndex = 0;
            for (let i = dataStart; i < lines.length && pointIndex < pointCount; i++) {
              const line = lines[i].trim();
              if (!line || line.length === 0) continue;
              
              const parts = line.split(/\s+/);
              if (parts.length >= 3) {
                const x = parseFloat(parts[0]);
                const y = parseFloat(parts[1]);
                const z = parseFloat(parts[2]);
                
                // Skip invalid points
                if (isNaN(x) || isNaN(y) || isNaN(z)) continue;
                
                // TruckScenes uses same coordinate system as nuScenes
                // X: forward, Y: left, Z: up
                // Convert to Three.js: X: right, Y: up, Z: backward
                points[pointIndex * 3] = -y;      // Y left → X right (negated)
                points[pointIndex * 3 + 1] = z;   // Z up → Y up
                points[pointIndex * 3 + 2] = -x;  // X forward → Z backward (negated)
                
                // Get intensity (field index varies, usually 3rd or 4th field)
                const intensityIdx = fields.indexOf('intensity');
                const intensity = intensityIdx >= 0 && parts.length > intensityIdx 
                  ? parseFloat(parts[intensityIdx]) 
                  : (parts.length > 3 ? parseFloat(parts[3]) : 50);
                
                // Color based on height and intensity
                const heightNorm = Math.max(0, Math.min(1, (z + 2) / 6));
                const intensityNorm = Math.max(0.3, Math.min(intensity / 100, 1)); // Ensure minimum brightness
                
                colors[pointIndex * 3] = intensityNorm * (1 - heightNorm);     // Red: ground
                colors[pointIndex * 3 + 1] = intensityNorm * heightNorm;       // Green: elevated
                colors[pointIndex * 3 + 2] = intensityNorm * 0.6;              // Blue: increased for visibility
                
                pointIndex++;
              }
            }
            console.log(`[LiDAR] Parsed ${pointIndex} valid points from ASCII PCD`);
            
            // Trim arrays if we parsed fewer points than expected
            if (pointIndex < pointCount) {
              points = points.slice(0, pointIndex * 3);
              colors = colors.slice(0, pointIndex * 3);
            }
          } else if (dataFormat === 'binary_compressed') {
            // Binary compressed PCD format - NOT SUPPORTED
            // TruckScenes uses LZF compression which requires decompression
            console.error(`[LiDAR] binary_compressed format is not supported in browser`);
            console.error(`[LiDAR] Please convert PCD to uncompressed format or use ASCII format`);
            throw new Error(
              'Compressed PCD files are not supported. ' +
              'Please convert to ASCII or uncompressed binary PCD format. ' +
              'You can use: pcl_convert_pcd_ascii_binary input.pcd output.pcd 0'
            );
          } else if (dataFormat === 'binary') {
            // Uncompressed binary PCD format
            // Find where binary data starts (after header)
            let headerBytes = 0;
            for (let i = 0; i < buffer.byteLength; i++) {
              if (i >= 10) {
                const check = String.fromCharCode(
                  new Uint8Array(buffer, i - 10, 10)[0],
                  new Uint8Array(buffer, i - 10, 10)[1],
                  new Uint8Array(buffer, i - 10, 10)[2],
                  new Uint8Array(buffer, i - 10, 10)[3],
                  new Uint8Array(buffer, i - 10, 10)[4]
                );
                if (check === 'DATA ') {
                  // Skip to end of line
                  let j = i;
                  while (j < buffer.byteLength && new Uint8Array(buffer)[j] !== 10) j++;
                  headerBytes = j + 1;
                  break;
                }
              }
            }
            
            console.log(`[LiDAR] Binary data starts at byte ${headerBytes}`);
            
            // Calculate stride (bytes per point)
            const stride = sizes.reduce((sum, size) => sum + size, 0);
            console.log(`[LiDAR] Point stride: ${stride} bytes`);
            console.log(`[LiDAR] Available data: ${buffer.byteLength - headerBytes} bytes`);
            console.log(`[LiDAR] Expected data: ${pointCount * stride} bytes`);
            
            const binaryData = new DataView(buffer, headerBytes);
            const xIdx = fields.indexOf('x');
            const yIdx = fields.indexOf('y');
            const zIdx = fields.indexOf('z');
            const intensityIdx = fields.indexOf('intensity');
            
            // Calculate actual number of points we can read
            const maxPoints = Math.floor((buffer.byteLength - headerBytes) / stride);
            const actualPoints = Math.min(pointCount, maxPoints);
            console.log(`[LiDAR] Will read ${actualPoints} points (max: ${maxPoints})`);
            
            let pointIndex = 0;
            for (let i = 0; i < actualPoints; i++) {
              try {
                const offset = i * stride;
                const x = binaryData.getFloat32(offset + xIdx * 4, true);
                const y = binaryData.getFloat32(offset + yIdx * 4, true);
                const z = binaryData.getFloat32(offset + zIdx * 4, true);
                const intensity = intensityIdx >= 0 
                  ? binaryData.getFloat32(offset + intensityIdx * 4, true) 
                  : 50;
                
                // Skip invalid points
                if (isNaN(x) || isNaN(y) || isNaN(z)) continue;
                
                // Convert coordinates (TruckScenes uses same as nuScenes)
                points[pointIndex * 3] = -y;      // Y left → X right (negated)
                points[pointIndex * 3 + 1] = z;   // Z up → Y up
                points[pointIndex * 3 + 2] = -x;  // X forward → Z backward (negated)
                
                const heightNorm = Math.max(0, Math.min(1, (z + 2) / 6));
                const intensityNorm = Math.max(0.3, Math.min(intensity / 100, 1)); // Ensure minimum brightness
                
                colors[pointIndex * 3] = intensityNorm * (1 - heightNorm);
                colors[pointIndex * 3 + 1] = intensityNorm * heightNorm;
                colors[pointIndex * 3 + 2] = intensityNorm * 0.6; // Increased for visibility
                
                pointIndex++;
              } catch (e) {
                console.warn(`[LiDAR] Error reading point ${i}:`, e);
                break;
              }
            }
            console.log(`[LiDAR] Parsed ${pointIndex} valid points from binary PCD`);
            
            // Trim arrays to actual size
            if (pointIndex < pointCount) {
              points = points.slice(0, pointIndex * 3);
              colors = colors.slice(0, pointIndex * 3);
            }
          }
        } else if (lidarPath.includes(".pcd.bin")) {
          // nuScenes format: each point is 5 floats (x, y, z, intensity, ring_index)
          const dataView = new DataView(buffer);
          const numPoints = buffer.byteLength / (5 * 4);
          points = new Float32Array(numPoints * 3);
          colors = new Float32Array(numPoints * 3);

          for (let i = 0; i < numPoints; i++) {
            // Read nuScenes format: X forward, Y left, Z up
            const x = dataView.getFloat32(i * 20, true);
            const y = dataView.getFloat32(i * 20 + 4, true);
            const z = dataView.getFloat32(i * 20 + 8, true);
            const intensity = dataView.getFloat32(i * 20 + 12, true);

            // Convert nuScenes (X forward, Y left, Z up) to Three.js (X right, Y up, Z backward)
            points[i * 3] = -y;  // nuScenes Y left → Three.js X right (negated)
            points[i * 3 + 1] = z;   // nuScenes Z up → Three.js Y up
            points[i * 3 + 2] = -x;  // nuScenes X forward → Three.js Z backward (negated)

            // Color based on height and intensity blend
            const heightNorm = Math.max(0, Math.min(1, (z + 2) / 6));
            const intensityNorm = Math.max(0.3, Math.min(intensity / 255, 1)); // Ensure minimum brightness

            colors[i * 3] = intensityNorm * (1 - heightNorm);     // Red: more at ground level
            colors[i * 3 + 1] = intensityNorm * heightNorm;       // Green: more at higher elevations
            colors[i * 3 + 2] = intensityNorm * 0.6;              // Blue: increased for better visibility
          }
        } else if (source?.toLowerCase() === 'once' && lidarPath.endsWith('.bin')) {
          // ONCE format: each point is 4 floats (x, y, z, intensity)
          const dataView = new DataView(buffer);
          const numPoints = buffer.byteLength / (4 * 4);
          console.log(`[LiDAR] ONCE format detected: ${numPoints} points`);
          
          points = new Float32Array(numPoints * 3);
          colors = new Float32Array(numPoints * 3);

          // ONCE flip transform to convert to nuScenes coordinate system
          // flip_transform = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
          // This transforms: ONCE (x, y, z) → nuScenes (-x, -y, z)
          
          for (let i = 0; i < numPoints; i++) {
            // Read ONCE format
            const x_once = dataView.getFloat32(i * 16, true);
            const y_once = dataView.getFloat32(i * 16 + 4, true);
            const z_once = dataView.getFloat32(i * 16 + 8, true);
            const intensity = dataView.getFloat32(i * 16 + 12, true);

            // Apply ONCE → nuScenes transform
            // [x_nu]   [-1  0  0] [x_once]   [-x_once]
            // [y_nu] = [0  -1  0] [y_once] = [-y_once]
            // [z_nu]   [0   0  1] [z_once]   [z_once]
            const x_nuscenes = -x_once;
            const y_nuscenes = -y_once;
            const z_nuscenes = z_once;

            // Convert nuScenes → Three.js (X right, Y up, Z backward)
            points[i * 3] = -y_nuscenes;      // nuScenes Y left → Three.js X right (negated)
            points[i * 3 + 1] = z_nuscenes - 0.5;  // nuScenes Z up → Three.js Y up
            points[i * 3 + 2] = -x_nuscenes;      // nuScenes X forward → Three.js Z backward (negated)

            // Color based on height and intensity blend
            const heightNorm = Math.max(0, Math.min(1, (z_nuscenes + 2) / 6));
            const intensityNorm = Math.max(0.3, Math.min(intensity / 255, 1)); // Ensure minimum brightness

            colors[i * 3] = intensityNorm * (1 - heightNorm);     // Red: more at ground level
            colors[i * 3 + 1] = intensityNorm * heightNorm;       // Green: more at higher elevations
            colors[i * 3 + 2] = intensityNorm * 0.6;              // Blue: increased for better visibility
          }
        } else if (source?.toLowerCase() === 'waymo' && lidarPath.endsWith('.bin')) {
          // Waymo format: each point is 4-6 floats (x, y, z, intensity, [elongation, [range]])
          // Auto-detect stride based on file size
          const dataView = new DataView(buffer);
          let stride = 4; // Default: 4 floats per point
          
          // Try to detect stride
          if (buffer.byteLength % (6 * 4) === 0) {
            stride = 6;
          } else if (buffer.byteLength % (5 * 4) === 0) {
            stride = 5;
          } else if (buffer.byteLength % (4 * 4) === 0) {
            stride = 4;
          }
          
          const numPoints = buffer.byteLength / (stride * 4);
          console.log(`[LiDAR] Waymo format detected: ${numPoints} points, stride: ${stride} floats`);
          
          points = new Float32Array(numPoints * 3);
          colors = new Float32Array(numPoints * 3);

          // Waymo coordinate system transformation
          // Apply 90-degree rotation around Z-axis to align with nuScenes
          // Waymo (x, y, z) → Rotated (-y, x, z)
          
          for (let i = 0; i < numPoints; i++) {
            // Read Waymo format
            const x_waymo = dataView.getFloat32(i * stride * 4, true);
            const y_waymo = dataView.getFloat32(i * stride * 4 + 4, true);
            const z_waymo = dataView.getFloat32(i * stride * 4 + 8, true);
            const intensity = dataView.getFloat32(i * stride * 4 + 12, true);

            // Apply 90-degree rotation: (x, y, z) → (-y, x, z)
            const x_rotated = -y_waymo;
            const y_rotated = x_waymo;
            const z_rotated = z_waymo;

            // Convert to Three.js (X right, Y up, Z backward)
            points[i * 3] = -y_rotated;       // Rotated Y left → Three.js X right (negated)
            points[i * 3 + 1] = z_rotated - 0.5;  // Rotated Z up → Three.js Y up
            points[i * 3 + 2] = -x_rotated;       // Rotated X forward → Three.js Z backward (negated)

            // Color based on height and intensity blend
            const heightNorm = Math.max(0, Math.min(1, (z_rotated + 2) / 6));
            // Waymo intensity is typically 0-255, but can vary
            const intensityNorm = Math.max(0.3, Math.min(intensity / 255, 1)); // Ensure minimum brightness

            colors[i * 3] = intensityNorm * (1 - heightNorm);     // Red: more at ground level
            colors[i * 3 + 1] = intensityNorm * heightNorm;       // Green: more at higher elevations
            colors[i * 3 + 2] = intensityNorm * 0.6;              // Blue: increased for better visibility
          }
        } else {
          // KITTI/Generic format: each point is 4 floats (x, y, z, intensity/reflectance)
          const dataView = new DataView(buffer);
          const numPoints = buffer.byteLength / (4 * 4);
          console.log(`[LiDAR] Generic binary format: ${numPoints} points`);
          points = new Float32Array(numPoints * 3);
          colors = new Float32Array(numPoints * 3);

          for (let i = 0; i < numPoints; i++) {
            const x = dataView.getFloat32(i * 16, true);
            const y = dataView.getFloat32(i * 16 + 4, true);
            const z = dataView.getFloat32(i * 16 + 8, true);
            const reflectance = dataView.getFloat32(i * 16 + 12, true);

            // KITTI uses similar coordinate system to nuScenes
            points[i * 3] = -y;
            points[i * 3 + 1] = z;
            points[i * 3 + 2] = -x;

            // Color based on height and reflectance blend
            const heightNorm = Math.max(0, Math.min(1, (z + 2) / 6));
            const refNorm = Math.max(0.3, Math.min(reflectance, 1)); // Ensure minimum brightness

            colors[i * 3] = refNorm * (1 - heightNorm);     // Red: more at ground level
            colors[i * 3 + 1] = refNorm * heightNorm;       // Green: more at higher elevations
            colors[i * 3 + 2] = refNorm * 0.6;              // Blue: increased for better visibility
          }
        }

        // Calculate stats
        const numPoints = points.length / 3;
        let minZ = Infinity, maxZ = -Infinity;
        for (let i = 0; i < numPoints; i++) {
          const z = points[i * 3 + 1];
          if (z < minZ) minZ = z;
          if (z > maxZ) maxZ = z;
        }
        setStats({ count: numPoints, minZ, maxZ });
        
        console.log(`[LiDAR] Rendering ${numPoints} points with colors`);
        console.log(`[LiDAR] Color sample (first point): RGB(${colors[0]?.toFixed(2)}, ${colors[1]?.toFixed(2)}, ${colors[2]?.toFixed(2)})`);

        // Create point cloud
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(points, 3));
        geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
          size: 0.15,  // Increased from 0.05 for better visibility
          vertexColors: true,
          sizeAttenuation: true,
          transparent: false,  // Changed to false for better color visibility
          opacity: 1.0,  // Full opacity
        });

        const pointCloud = new THREE.Points(geometry, material);
        
        // Check if we've been cleaned up before adding to scene
        if (isCleanedUp || !scene) {
          console.log('[LiDAR] Cleanup occurred, skipping point cloud addition');
          geometry.dispose();
          material.dispose();
          return;
        }
        
        scene.add(pointCloud);
        console.log(`[LiDAR] Point cloud added to scene with vertex colors enabled`);
        pointCloudLoaded = true;

        } catch (lidarError) {
          // Point cloud loading failed - log warning but continue with annotations
          console.warn('[LiDAR] Failed to load point cloud, showing only 3D annotations:', lidarError);
          // setError(lidarError instanceof Error ? `Point cloud unavailable: ${lidarError.message}` : 'Point cloud unavailable');
          setStats({ count: 0, minZ: 0, maxZ: 0 });
          // Don't throw - continue to load annotations
        }

        // Load annotations (always shown, even if point cloud fails)
        if (isCleanedUp || !scene) {
          console.log('[LiDAR] Cleanup occurred, skipping annotations');
          return;
        }
        
        if (annotations && showAnnotations && annotations.bbox_3d.length > 0) {
          /**
           * Generate consistent color from track ID
           */
          const getColorFromId = (trackId: string): number => {
            let hash = 0;
            for (let i = 0; i < trackId.length; i++) {
              hash = trackId.charCodeAt(i) + ((hash << 5) - hash);
            }
            // Generate bright, saturated colors
            const h = Math.abs(hash % 360);
            const s = 70 + (Math.abs(hash) % 30); // 70-100% saturation
            const l = 50 + (Math.abs(hash >> 8) % 20); // 50-70% lightness
            
            // Convert HSL to RGB
            const hslToRgb = (h: number, s: number, l: number): number => {
              s /= 100;
              l /= 100;
              const k = (n: number) => (n + h / 30) % 12;
              const a = s * Math.min(l, 1 - l);
              const f = (n: number) =>
                l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)));
              const r = Math.round(255 * f(0));
              const g = Math.round(255 * f(8));
              const b = Math.round(255 * f(4));
              return (r << 16) | (g << 8) | b;
            };
            
            return hslToRgb(h, s, l);
          };

          // Add ego vehicle to annotations as a normal object
          // Ego vehicle: track_id "00000", at origin (0, 0, 0)
          // Calculate ego yaw from ego_pose if available
          let egoYaw = 1.5708; // Default to π/2 (90 degrees)
          if (calibration?.ego_pose && Array.isArray(calibration.ego_pose) && calibration.ego_pose.length >= 2) {
            const egoPose = calibration.ego_pose;
            // ego_yaw = arctan2(ego_pose[1, 0], ego_pose[0, 0])
            // In JavaScript: ego_pose[1][0] is row 1, col 0; ego_pose[0][0] is row 0, col 0
            egoYaw = Math.atan2(egoPose[1][0], egoPose[0][0]);
          }
          const egoVehicle = {
            bbox: [0, 0, 0, 4.5, 1.8, 1.5, Math.PI / 2  ], // x, y, z, length, width, height, yaw (nuScenes coords)
            label: "ego",
            trackId: "00000"
          };
          
          // Create extended arrays with ego vehicle prepended
          const extendedBboxes = [egoVehicle.bbox, ...(annotations.bbox_3d || [])];
          const extendedLabels = [egoVehicle.label, ...(annotations.label_3d || [])];
          const extendedTrackIds = [egoVehicle.trackId, ...(annotations.track_3d || [])];

          for (let i = 0; i < extendedBboxes.length; i++) {
            const bbox = extendedBboxes[i];
            const label = extendedLabels[i] || "unknown";
            const trackId = extendedTrackIds[i] || "";

            // Convert bbox from nuScenes to Three.js coordinates
            const [x, y, z, width, length, height, rotation] = bbox;
            // Step 1: Calculate rotated corners in nuScenes coordinate system
            const halfWidth = width / 2;
            const halfLength = length / 2;
            const halfHeight = height / 2;
            
            // Rotation matrix around Z-axis (yaw) in nuScenes
            const cosYaw = Math.cos(rotation);
            const sinYaw = Math.sin(rotation);
            
            // Step 2: Calculate front center point in nuScenes (for direction line)
            // Front center in local frame: (+halfWidth, 0, 0) - positive X is forward in nuScenes
            const frontLocalX = halfWidth;
            const frontLocalY = 0;
            const frontLocalZ = 0;
            
            // Apply rotation in nuScenes (rotation around Z-axis, so Z doesn't change)
            const frontRotatedX = frontLocalX * cosYaw - frontLocalY * sinYaw;
            const frontRotatedY = frontLocalX * sinYaw + frontLocalY * cosYaw;
            const frontRotatedZ = frontLocalZ;  // Z unchanged by yaw rotation
            
            // Translate to world coordinates in nuScenes
            const frontWorldX = x + frontRotatedX;
            const frontWorldY = y + frontRotatedY;
            const frontWorldZ = z + frontRotatedZ;
            
            // Step 3: Transform center and front to Three.js coordinates
            const threeX = -y;   // Center
            const threeY = z;
            const threeZ = -x;
            
            const frontThreeX = -frontWorldY;
            const frontThreeY = frontWorldZ;
            const frontThreeZ = -frontWorldX;

            // Get color based on track ID and selection
            let color: number;
            if ((showRelationships || showInternalInteractions || showExternalInteractions || showOcclusions) && selectedTrackId !== "all") {
              // When filtering by track ID (in any interaction mode): red for selected, blue for others
              color = trackId === selectedTrackId ? 0xff0000 : 0x0088ff;
            } else {
              // Normal mode: color based on track ID (ego included)
              color = trackId ? getColorFromId(trackId) : 0xffffff;
            }
            
            // Step 4: Create rotated bounding box by calculating all 8 corners
            // In nuScenes: length is Y-axis extent, width is X-axis extent
            // Define corners in local box frame (before rotation)
            const localCorners = [
              [-halfWidth, -halfLength, -halfHeight], // 0: back-right-bottom
              [halfWidth, -halfLength, -halfHeight],  // 1: front-right-bottom
              [halfWidth, halfLength, -halfHeight],   // 2: front-left-bottom
              [-halfWidth, halfLength, -halfHeight],  // 3: back-left-bottom
              [-halfWidth, -halfLength, halfHeight],  // 4: back-right-top
              [halfWidth, -halfLength, halfHeight],   // 5: front-right-top
              [halfWidth, halfLength, halfHeight],    // 6: front-left-top
              [-halfWidth, halfLength, halfHeight],   // 7: back-left-top
            ];
            
            // Rotate each corner in nuScenes, then transform to Three.js
            const threeCorners = localCorners.map(([lx, ly, lz]) => {
              // Apply rotation around Z-axis in nuScenes
              const rotX = lx * cosYaw - ly * sinYaw;
              const rotY = lx * sinYaw + ly * cosYaw;
              const rotZ = lz;
              
              // Translate to world position
              const wx = x + rotX;
              const wy = y + rotY;
              const wz = z + rotZ;
              
              // Transform to Three.js
              return new THREE.Vector3(-wy, wz, -wx);
            });
            
            // Create line segments connecting the corners
            const vertices = new Float32Array([
              // Bottom face (4 edges)
              ...threeCorners[0].toArray(), ...threeCorners[1].toArray(),
              ...threeCorners[1].toArray(), ...threeCorners[2].toArray(),
              ...threeCorners[2].toArray(), ...threeCorners[3].toArray(),
              ...threeCorners[3].toArray(), ...threeCorners[0].toArray(),
              // Top face (4 edges)
              ...threeCorners[4].toArray(), ...threeCorners[5].toArray(),
              ...threeCorners[5].toArray(), ...threeCorners[6].toArray(),
              ...threeCorners[6].toArray(), ...threeCorners[7].toArray(),
              ...threeCorners[7].toArray(), ...threeCorners[4].toArray(),
              // Vertical edges (4 edges)
              ...threeCorners[0].toArray(), ...threeCorners[4].toArray(),
              ...threeCorners[1].toArray(), ...threeCorners[5].toArray(),
              ...threeCorners[2].toArray(), ...threeCorners[6].toArray(),
              ...threeCorners[3].toArray(), ...threeCorners[7].toArray(),
            ]);
            
            const boxGeometry = new THREE.BufferGeometry();
            boxGeometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
            
            const lineMaterial = new THREE.LineBasicMaterial({ 
              color,
              linewidth: selectedTrackId === trackId ? 4 : 3,
              transparent: false,
              opacity: 1.0,
            });
            
            const box = new THREE.LineSegments(boxGeometry, lineMaterial);
            annotationGroup.add(box);

            // Step 5: Add direction line from center bottom to front center bottom
            const startPoint = new THREE.Vector3(threeX, threeY - halfHeight, threeZ);
            const endPoint = new THREE.Vector3(frontThreeX, frontThreeY - halfHeight, frontThreeZ);
            
            // Create direction line
            const directionGeometry = new THREE.BufferGeometry().setFromPoints([startPoint, endPoint]);
            const directionMaterial = new THREE.LineBasicMaterial({
              color,
              linewidth: 3,
              transparent: false,
              opacity: 1.0,
            });
            const directionLine = new THREE.Line(directionGeometry, directionMaterial);
            annotationGroup.add(directionLine);

            // Add label with class name and first 5 chars of track ID
            const shortTrackId = trackId.substring(0, 5);
            const labelText = `${label} ${shortTrackId}`;
            
            const canvas = document.createElement("canvas");
            const context = canvas.getContext("2d");
            if (context) {
              canvas.width = 256;
              canvas.height = 64;
              context.fillStyle = "rgba(0, 0, 0, 0.7)";
              context.fillRect(0, 0, canvas.width, canvas.height);
              context.font = "20px Arial";
              context.fillStyle = "white";
              context.fillText(labelText, 10, 35);

              const texture = new THREE.CanvasTexture(canvas);
              const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
              const sprite = new THREE.Sprite(spriteMaterial);
              sprite.position.set(threeX, threeY + height / 2 + 1, threeZ);
              sprite.scale.set(4, 1, 1);
              annotationGroup.add(sprite);
            }
          }
        }

        // Render internal interactions (actions) for each object
        if (showInternalInteractions && internalInteractions && annotations && annotations.bbox_3d) {
          for (const interaction of internalInteractions) {
            // Find the corresponding annotation (including ego at index 0)
            const extendedTrackIds = ["00000", ...(annotations.track_3d || [])];
            const extendedBboxes = [[0, 0, 0, 4.5, 1.8, 1.5, 0], ...(annotations.bbox_3d || [])];
            
            const objIndex = extendedTrackIds.indexOf(interaction.track_id);
            if (objIndex === undefined || objIndex < 0) continue;
            
            const bbox = extendedBboxes[objIndex];
            const [x, y, z, width, length, height] = bbox;
            
            // Transform to Three.js coordinates
            const threeX = -y;
            const threeY = z;
            const threeZ = -x;
            
            // Create action badges (colored pills) instead of single text
            const actions = interaction.actions.length > 0 ? interaction.actions : ['No actions'];
            
            const actionCanvas = document.createElement("canvas");
            const actionContext = actionCanvas.getContext("2d");
            if (actionContext) {
              // Calculate canvas size based on number of actions
              const badgeHeight = 28;
              const badgeSpacing = 6;
              const canvasWidth = 450;
              const canvasHeight = Math.max(80, actions.length * (badgeHeight + badgeSpacing) + 20);
              
              actionCanvas.width = canvasWidth;
              actionCanvas.height = canvasHeight;
              
              // Semi-transparent background
              actionContext.fillStyle = "rgba(0, 0, 0, 0.75)";
              actionContext.fillRect(0, 0, canvasWidth, canvasHeight);
              
              // Draw each action as a colored badge
              let yPos = 15;
              for (const action of actions) {
                const actionText = action.replace(/_/g, ' ').toUpperCase();
                const color = actionColors[action] || '#6B7280'; // Default gray
                
                // Draw rounded rectangle background
                actionContext.fillStyle = color;
                actionContext.beginPath();
                actionContext.roundRect(10, yPos, canvasWidth - 20, badgeHeight, 4);
                actionContext.fill();
                
                // Draw text
                actionContext.font = "Bold 16px Arial";
                actionContext.fillStyle = "white";
                actionContext.textAlign = "center";
                actionContext.textBaseline = "middle";
                actionContext.fillText(actionText, canvasWidth / 2, yPos + badgeHeight / 2);
                
                yPos += badgeHeight + badgeSpacing;
              }

              const actionTexture = new THREE.CanvasTexture(actionCanvas);
              const actionSpriteMaterial = new THREE.SpriteMaterial({ 
                map: actionTexture,
                depthTest: false // Render on top
              });
              const actionSprite = new THREE.Sprite(actionSpriteMaterial);
              actionSprite.position.set(threeX, threeY + height / 2 + 2.5, threeZ); // Above object label
              
              // Scale based on number of actions
              const scaleY = canvasHeight / 64; // Adjust vertical scale
              actionSprite.scale.set(7, scaleY * 1.2, 1);
              
              // Store track_id for raycasting
              actionSprite.userData = { 
                type: 'action', 
                track_id: interaction.track_id,
                label: interaction.label,
                actions: interaction.actions
              };
              
              annotationGroup.add(actionSprite);
            }
            
            // Visualize velocity vector if available in details
            if (interaction.details && (interaction.details.velocity_x !== undefined || interaction.details.velocity_y !== undefined)) {
              const velX = interaction.details.velocity_x || 0;
              const velY = interaction.details.velocity_y || 0;
              const velMagnitude = interaction.details.velocity_magnitude || Math.sqrt(velX * velX + velY * velY);
              
              // Skip if velocity is too small
              if (velMagnitude < 0.1) continue;
              
              // Normalize and scale velocity vector for visualization
              // Scale down the magnitude for reasonable arrow size
              const scaleFactor = Math.min(5.0, velMagnitude / 100); // Scale based on magnitude
              
              // In nuScenes coordinate system: X forward, Y left
              // Transform to Three.js: X right, Y up, Z backward
              // velocity_x (forward) → -Z in Three.js
              // velocity_y (left) → -X in Three.js (negated because X is right)
              const arrowStartX = threeX;
              const arrowStartY = threeY;
              const arrowStartZ = threeZ;
              
              // Apply coordinate transformation for velocity direction
              const velocityThreeX = -velY * scaleFactor / velMagnitude; // Y left → -X right
              const velocityThreeZ = -velX * scaleFactor / velMagnitude; // X forward → -Z backward
              
              const arrowEndX = arrowStartX + velocityThreeX;
              const arrowEndY = arrowStartY; // No vertical component
              const arrowEndZ = arrowStartZ + velocityThreeZ;
              
              // Create arrow shaft (line)
              const arrowStart = new THREE.Vector3(arrowStartX, arrowStartY, arrowStartZ);
              const arrowEnd = new THREE.Vector3(arrowEndX, arrowEndY, arrowEndZ);
              
              const arrowGeometry = new THREE.BufferGeometry().setFromPoints([arrowStart, arrowEnd]);
              const arrowMaterial = new THREE.LineBasicMaterial({
                color: 0xFFFF00, // Yellow for velocity vectors
                linewidth: 3,
                transparent: false,
                opacity: 1.0,
              });
              const arrowLine = new THREE.Line(arrowGeometry, arrowMaterial);
              annotationGroup.add(arrowLine);
              
              // Create arrow head (cone)
              const arrowHeadLength = 0.5;
              const arrowHeadRadius = 0.2;
              const arrowHeadGeometry = new THREE.ConeGeometry(arrowHeadRadius, arrowHeadLength, 8);
              const arrowHeadMaterial = new THREE.MeshBasicMaterial({ 
                color: 0xFFFF00,
                transparent: false,
                opacity: 1.0
              });
              const arrowHead = new THREE.Mesh(arrowHeadGeometry, arrowHeadMaterial);
              
              // Position arrow head at the end
              arrowHead.position.copy(arrowEnd);
              
              // Orient arrow head to point in velocity direction
              const direction = new THREE.Vector3().subVectors(arrowEnd, arrowStart).normalize();
              const upVector = new THREE.Vector3(0, 1, 0);
              const quaternion = new THREE.Quaternion();
              quaternion.setFromUnitVectors(upVector, direction);
              arrowHead.quaternion.copy(quaternion);
              
              annotationGroup.add(arrowHead);
              
              // Add velocity magnitude label
              const velCanvas = document.createElement("canvas");
              const velContext = velCanvas.getContext("2d");
              if (velContext) {
                const velText = `${velMagnitude.toFixed(1)} m/s`;
                velCanvas.width = 200;
                velCanvas.height = 48;
                
                // Background
                velContext.fillStyle = "rgba(255, 255, 0, 0.9)";
                velContext.fillRect(0, 0, velCanvas.width, velCanvas.height);
                
                // Text
                velContext.font = "Bold 16px Arial";
                velContext.fillStyle = "black";
                velContext.textAlign = "center";
                velContext.textBaseline = "middle";
                velContext.fillText(velText, velCanvas.width / 2, velCanvas.height / 2);
                
                const velTexture = new THREE.CanvasTexture(velCanvas);
                const velSpriteMaterial = new THREE.SpriteMaterial({ map: velTexture });
                const velSprite = new THREE.Sprite(velSpriteMaterial);
                
                // Position label at the midpoint of the arrow
                const midX = (arrowStartX + arrowEndX) / 2;
                const midY = (arrowStartY + arrowEndY) / 2 + 0.5; // Slightly above
                const midZ = (arrowStartZ + arrowEndZ) / 2;
                velSprite.position.set(midX, midY, midZ);
                velSprite.scale.set(2, 0.5, 1);
                
                annotationGroup.add(velSprite);
              }
            }
          }
        }

        // Position camera behind selected object (only if selection changed)
        if (selectedTrackId !== "all" && annotations && annotations.bbox_3d) {
          // Only reposition camera if:
          // 1. No saved state exists, OR
          // 2. Record changed, OR
          // 3. Selected track ID changed (user clicked Next/Prev object)
          const shouldRepositionCamera = !cameraStateRef.current || 
                                         cameraStateRef.current.recordId !== recordId ||
                                         cameraStateRef.current.selectedTrackId !== selectedTrackId;
          
          if (shouldRepositionCamera) {
            // Find the selected object
            const selectedIndex = annotations.track_3d?.indexOf(selectedTrackId);
            if (selectedIndex !== undefined && selectedIndex >= 0) {
              const bbox = annotations.bbox_3d[selectedIndex];
              const [x, y, z] = bbox;
              
              // Transform to Three.js coordinates
              const threeX = -y;
              const threeY = z;
              const threeZ = -x;
              
              // Position camera: object position + offset (15, 5, 0) in LiDAR coordinates
              const cameraX = threeX;           // No lateral offset
              const cameraY = threeY + 5;       // +5 up
              const cameraZ = threeZ + 15;      // +15 behind (in Three.js -Z direction)
              
              camera.position.set(cameraX, cameraY, cameraZ);
              controls.target.set(threeX, threeY, threeZ);
              controls.update();
              
              // Save this as the new camera position with the selected track ID
              cameraStateRef.current = {
                position: { x: cameraX, y: cameraY, z: cameraZ },
                target: { x: threeX, y: threeY, z: threeZ },
                recordId: recordId,
                selectedTrackId: selectedTrackId
              };
              
              console.log(`Camera repositioned for new target object ${selectedTrackId.substring(0, 8)}...`);
              console.log(`Position: (${cameraX.toFixed(2)}, ${cameraY.toFixed(2)}, ${cameraZ.toFixed(2)})`);
            }
          } else {
            console.log(`Camera position preserved (same object: ${selectedTrackId.substring(0, 8)}...)`);
          }
        }

        // Visualize spatial relationships if enabled
        if (showRelationships && spatialRelations && spatialRelations.length > 0) {
          console.log(`Loading ${spatialRelations.length} spatial relationships`);
          
          // Build track ID to position map from annotations
          const trackPositions = new Map<string, any>();
          if (annotations && annotations.bbox_3d) {
            for (let i = 0; i < annotations.bbox_3d.length; i++) {
              const bbox = annotations.bbox_3d[i];
              const trackId = annotations.track_3d?.[i];
              if (trackId) {
                const [x, y, z] = bbox;
                // Transform to Three.js coordinates
                trackPositions.set(trackId, new THREE.Vector3(-y, z, -x));
              }
            }
          }
          
          // Add ego vehicle at origin (0, 0, 0) in Three.js coordinates
          trackPositions.set("00000", new THREE.Vector3(0, 0, 0));

          // Group relationships by source-target pair to offset multiple relations
          const relationshipPairs = new Map<string, Array<{
            source: any;
            target: any;
            relations: string[];
            sourceLabel: string;
            targetLabel: string;
          }>>();

          for (const rel of spatialRelations) {
            // Filter: only show relationships where source matches selectedTrackId (if not "all")
            if (selectedTrackId !== "all" && rel.source_object.track_id !== selectedTrackId) {
              continue;
            }

            const sourcePos = trackPositions.get(rel.source_object.track_id);
            const targetPos = trackPositions.get(rel.target_object.track_id);

            if (sourcePos && targetPos) {
              const pairKey = `${rel.source_object.track_id}_${rel.target_object.track_id}`;
              
              if (!relationshipPairs.has(pairKey)) {
                relationshipPairs.set(pairKey, []);
              }
              
              relationshipPairs.get(pairKey)!.push({
                source: sourcePos.clone(),
                target: targetPos.clone(),
                relations: rel.relations,
                sourceLabel: rel.source_object.label,
                targetLabel: rel.target_object.label,
              });
            }
          }

          // Draw curved lines for each relationship group
          relationshipPairs.forEach((rels, pairKey) => {
            rels.forEach((rel, idx) => {
              // Calculate curve offset based on index to avoid overlap
              const curveOffset = (idx - (rels.length - 1) / 2) * 0.5; // Offset perpendicular to line
              
              const midPoint = new THREE.Vector3()
                .addVectors(rel.source, rel.target)
                .multiplyScalar(0.5);

              // Calculate perpendicular vector for offset
              const direction = new THREE.Vector3().subVectors(rel.target, rel.source);
              const perpendicular = new THREE.Vector3(-direction.z, 0, direction.x).normalize();
              const controlPoint = midPoint.clone().add(perpendicular.multiplyScalar(curveOffset + 2));

              // Create quadratic bezier curve
              const curve = new THREE.QuadraticBezierCurve3(
                rel.source,
                controlPoint,
                rel.target
              );

              const points = curve.getPoints(20);
              const curveGeometry = new THREE.BufferGeometry().setFromPoints(points);
              
              // Use semi-transparent cyan for relationship lines
              const curveMaterial = new THREE.LineBasicMaterial({
                color: 0x00ffff,
                transparent: true,
                opacity: 0.4,
                linewidth: 2,
              });

              const curveLine = new THREE.Line(curveGeometry, curveMaterial);
              
              // Store metadata for interaction
              const [sourceTrackId, targetTrackId] = pairKey.split("_");
              (curveLine as any).userData = {
                type: 'relationship',
                source_track_id: sourceTrackId,
                target_track_id: targetTrackId,
                source_label: rel.sourceLabel,
                target_label: rel.targetLabel,
                relations: [...rel.relations],
                controlPoint: controlPoint.clone()
              };
              
              annotationGroup.add(curveLine);

              // Add label at midpoint with relation names
              const relationText = rel.relations.join(", ");
              const labelCanvas = document.createElement("canvas");
              const labelContext = labelCanvas.getContext("2d");
              
              if (labelContext) {
                // Bigger canvas and font when specific target is selected
                const isTargetSelected = selectedTrackId !== "all";
                const canvasWidth = isTargetSelected ? 384 : 256;
                const canvasHeight = isTargetSelected ? 72 : 48;
                const fontSize = isTargetSelected ? 20 : 14;
                const textY = isTargetSelected ? 42 : 28;
                
                labelCanvas.width = canvasWidth;
                labelCanvas.height = canvasHeight;
                labelContext.fillStyle = "rgba(0, 0, 0, 0.8)";
                labelContext.fillRect(0, 0, labelCanvas.width, labelCanvas.height);
                labelContext.font = `bold ${fontSize}px Arial`;
                labelContext.fillStyle = "#00ffff";
                labelContext.fillText(relationText, 10, textY);

                const labelTexture = new THREE.CanvasTexture(labelCanvas);
                const labelSpriteMaterial = new THREE.SpriteMaterial({ map: labelTexture });
                const labelSprite = new THREE.Sprite(labelSpriteMaterial);
                labelSprite.position.copy(controlPoint);
                
                // Store same metadata on sprite for clicking
                (labelSprite as any).userData = {
                  type: 'relationship',
                  source_track_id: sourceTrackId,
                  target_track_id: targetTrackId,
                  source_label: rel.sourceLabel,
                  target_label: rel.targetLabel,
                  relations: [...rel.relations],
                  controlPoint: controlPoint.clone()
                };
                
                // Bigger sprite scale when specific target is selected
                if (isTargetSelected) {
                  labelSprite.scale.set(5, 1, 1);
                } else {
                  labelSprite.scale.set(3, 0.6, 1);
                }
                
                annotationGroup.add(labelSprite);
              }
            });
          });

          console.log(`Rendered ${relationshipPairs.size} relationship groups`);
        }

        // Visualize external interactions if enabled
        if (showExternalInteractions && externalInteractions && externalInteractions.length > 0) {
          console.log(`[LidarViewer] Loading ${externalInteractions.length} external interactions`);
          
          // Build track ID to position map from annotations
          const trackPositions = new Map<string, any>();
          if (annotations && annotations.bbox_3d) {
            for (let i = 0; i < annotations.bbox_3d.length; i++) {
              const bbox = annotations.bbox_3d[i];
              const trackId = annotations.track_3d?.[i];
              if (trackId) {
                const [x, y, z] = bbox;
                // Transform to Three.js coordinates
                trackPositions.set(trackId, new THREE.Vector3(-y, z, -x));
              }
            }
          }
          
          // Add ego vehicle at origin (0, 0, 0) in Three.js coordinates
          trackPositions.set("00000", new THREE.Vector3(0, 0, 0));

          // Group interactions by source-target pair to offset multiple interactions
          const interactionPairs = new Map<string, Array<{
            source: any;
            target: any;
            interactions: string[];
            sourceLabel: string;
            targetLabel: string;
          }>>();

          for (const interaction of externalInteractions) {
            // Filter: only show interactions where source matches selectedTrackId (if not "all")
            if (selectedTrackId !== "all" && interaction.source_object.track_id !== selectedTrackId) {
              continue;
            }

            const sourcePos = trackPositions.get(interaction.source_object.track_id);
            const targetPos = trackPositions.get(interaction.target_object.track_id);

            if (sourcePos && targetPos && interaction.interaction_type) {
              const pairKey = `${interaction.source_object.track_id}_${interaction.target_object.track_id}`;
              
              if (!interactionPairs.has(pairKey)) {
                interactionPairs.set(pairKey, []);
              }
              
              interactionPairs.get(pairKey)!.push({
                source: sourcePos.clone(),
                target: targetPos.clone(),
                interactions: [interaction.interaction_type], // Wrap in array for consistent rendering
                sourceLabel: interaction.source_object.label,
                targetLabel: interaction.target_object.label,
              });
            }
          }

          // Draw lines for each interaction group (use different color than spatial relations)
          interactionPairs.forEach((interactions, pairKey) => {
            interactions.forEach((interaction, idx) => {
              // Calculate curve offset based on index to avoid overlap
              const curveOffset = (idx - (interactions.length - 1) / 2) * 0.5;
              
              const midPoint = new THREE.Vector3()
                .addVectors(interaction.source, interaction.target)
                .multiplyScalar(0.5);

              // Calculate perpendicular vector for offset
              const direction = new THREE.Vector3().subVectors(interaction.target, interaction.source);
              const perpendicular = new THREE.Vector3(-direction.z, 0, direction.x).normalize();
              const controlPoint = midPoint.clone().add(perpendicular.multiplyScalar(curveOffset + 2));

              // Create quadratic bezier curve
              const curve = new THREE.QuadraticBezierCurve3(
                interaction.source,
                controlPoint,
                interaction.target
              );

              const points = curve.getPoints(20);
              const curveGeometry = new THREE.BufferGeometry().setFromPoints(points);
              
              // Get interaction type and its color
              const interactionType = interaction.interactions[0] || 'unknown';
              const colorHex = externalInteractionColors[interactionType] || '#00ff88';
              const color = parseInt(colorHex.replace('#', '0x'));
              
              // Use color based on interaction type
              const curveMaterial = new THREE.LineBasicMaterial({
                color: color,
                transparent: true,
                opacity: 0.6,
                linewidth: 2,
              });

              const curveLine = new THREE.Line(curveGeometry, curveMaterial);
              
              // Store metadata for interaction
              const [sourceTrackId, targetTrackId] = pairKey.split("_");
              (curveLine as any).userData = {
                type: 'externalInteraction',
                source_track_id: sourceTrackId,
                target_track_id: targetTrackId,
                source_label: interaction.sourceLabel,
                target_label: interaction.targetLabel,
                interactions: [...interaction.interactions],
                controlPoint: controlPoint.clone()
              };
              
              annotationGroup.add(curveLine);
              
              // Add arrow at the target end to show direction (source -> target)
              // Get the last two points to calculate arrow direction
              const arrowStart = points[points.length - 2];
              const arrowEnd = points[points.length - 1]; // This is the target position
              
              // Calculate direction vector from second-to-last point to target
              const arrowDirection = new THREE.Vector3()
                .subVectors(arrowEnd, arrowStart)
                .normalize();
              
              // Create arrow cone
              const arrowLength = 0.8;
              const arrowRadius = 0.3;
              const arrowGeometry = new THREE.ConeGeometry(arrowRadius, arrowLength, 8);
              const arrowMaterial = new THREE.MeshBasicMaterial({ 
                color: color,
                transparent: true,
                opacity: 0.8
              });
              const arrowCone = new THREE.Mesh(arrowGeometry, arrowMaterial);
              
              // Position the arrow at the target
              arrowCone.position.copy(arrowEnd);
              
              // Orient the arrow to point in the direction of the curve
              // Default cone points up (0, 1, 0), we need to rotate it to point along arrowDirection
              const upVector = new THREE.Vector3(0, 1, 0);
              const quaternion = new THREE.Quaternion();
              quaternion.setFromUnitVectors(upVector, arrowDirection);
              arrowCone.quaternion.copy(quaternion);
              
              // Move the arrow back slightly so the tip is at the target
              arrowCone.position.add(arrowDirection.clone().multiplyScalar(-arrowLength / 2));
              
              // Store metadata on arrow too for raycasting
              (arrowCone as any).userData = {
                type: 'externalInteraction',
                source_track_id: sourceTrackId,
                target_track_id: targetTrackId,
                source_label: interaction.sourceLabel,
                target_label: interaction.targetLabel,
                interactions: [...interaction.interactions],
                controlPoint: controlPoint.clone()
              };
              
              annotationGroup.add(arrowCone);

              // Add label at midpoint with interaction name
              const interactionText = interactionType.replace(/_/g, ' ');
              const labelCanvas = document.createElement("canvas");
              const labelContext = labelCanvas.getContext("2d");
              
              if (labelContext) {
                // Bigger canvas and font when specific target is selected
                const isTargetSelected = selectedTrackId !== "all";
                const canvasWidth = isTargetSelected ? 384 : 256;
                const canvasHeight = isTargetSelected ? 72 : 48;
                const fontSize = isTargetSelected ? 20 : 14;
                const textY = isTargetSelected ? 42 : 28;
                
                labelCanvas.width = canvasWidth;
                labelCanvas.height = canvasHeight;
                
                // Use hex color for label background (convert to rgba)
                const r = parseInt(colorHex.slice(1, 3), 16);
                const g = parseInt(colorHex.slice(3, 5), 16);
                const b = parseInt(colorHex.slice(5, 7), 16);
                labelContext.fillStyle = `rgba(${r}, ${g}, ${b}, 0.9)`;
                
                labelContext.fillRect(0, 0, canvasWidth, canvasHeight);
                labelContext.font = `Bold ${fontSize}px Arial`;
                labelContext.fillStyle = "white";
                labelContext.textAlign = "center";
                labelContext.fillText(interactionText.toUpperCase(), canvasWidth / 2, textY);

                const labelTexture = new THREE.CanvasTexture(labelCanvas);
                const labelSpriteMaterial = new THREE.SpriteMaterial({ map: labelTexture });
                const labelSprite = new THREE.Sprite(labelSpriteMaterial);
                labelSprite.position.copy(controlPoint);
                
                // Store same metadata on sprite for clicking
                (labelSprite as any).userData = {
                  type: 'externalInteraction',
                  source_track_id: sourceTrackId,
                  target_track_id: targetTrackId,
                  source_label: interaction.sourceLabel,
                  target_label: interaction.targetLabel,
                  interactions: [...interaction.interactions],
                  controlPoint: controlPoint.clone()
                };
                
                // Bigger sprite scale when specific target is selected
                if (isTargetSelected) {
                  labelSprite.scale.set(5, 1, 1);
                } else {
                  labelSprite.scale.set(3, 0.6, 1);
                }
                
                annotationGroup.add(labelSprite);
              }
            });
          });

          console.log(`Rendered ${interactionPairs.size} external interaction groups`);
        }

        // Visualize occlusions if enabled
        if (showOcclusions && occlusions && occlusions.length > 0) {
          console.log(`[LidarViewer] Loading ${occlusions.length} occlusions`);
          
          // Build track ID to position map from annotations
          const trackPositions = new Map<string, any>();
          if (annotations && annotations.bbox_3d) {
            for (let i = 0; i < annotations.bbox_3d.length; i++) {
              const bbox = annotations.bbox_3d[i];
              const trackId = annotations.track_3d?.[i];
              if (trackId) {
                const [x, y, z] = bbox;
                // Transform to Three.js coordinates
                trackPositions.set(trackId, new THREE.Vector3(-y, z, -x));
              }
            }
          }
          
          // Add ego vehicle at origin (0, 0, 0) in Three.js coordinates
          trackPositions.set("00000", new THREE.Vector3(0, 0, 0));

          // Visibility color map (1-4 scale, where 1 is most occluded, 4 is fully visible)
          const visibilityColors: { [key: number]: string } = {
            1: '#EF4444', // Red - heavily occluded
            2: '#F59E0B', // Orange - partially occluded
            3: '#3B82F6', // Blue - mostly visible
            4: '#10B981', // Green - fully visible
          };

          // Group occlusions by source-target pair
          const occlusionPairs = new Map<string, Array<{
            source: any;
            target: any;
            visibility: number;
            sourceLabel: string;
            targetLabel: string;
          }>>();

          for (const occlusion of occlusions) {
            // Filter: only show occlusions where source matches selectedTrackId (if not "all")
            if (selectedTrackId !== "all" && occlusion.source_object.track_id !== selectedTrackId) {
              continue;
            }

            const sourcePos = trackPositions.get(occlusion.source_object.track_id);
            const targetPos = trackPositions.get(occlusion.target_object.track_id);

            if (sourcePos && targetPos) {
              const pairKey = `${occlusion.source_object.track_id}_${occlusion.target_object.track_id}`;
              
              if (!occlusionPairs.has(pairKey)) {
                occlusionPairs.set(pairKey, []);
              }
              
              occlusionPairs.get(pairKey)!.push({
                source: sourcePos.clone(),
                target: targetPos.clone(),
                visibility: occlusion.visibility,
                sourceLabel: occlusion.source_object.label,
                targetLabel: occlusion.target_object.label,
              });
            }
          }

          // Draw lines for each occlusion
          occlusionPairs.forEach((occs, pairKey) => {
            occs.forEach((occ, idx) => {
              // Calculate curve offset based on index to avoid overlap
              const curveOffset = (idx - (occs.length - 1) / 2) * 0.5;
              
              const midPoint = new THREE.Vector3()
                .addVectors(occ.source, occ.target)
                .multiplyScalar(0.5);

              // Calculate perpendicular vector for offset
              const direction = new THREE.Vector3().subVectors(occ.target, occ.source);
              const perpendicular = new THREE.Vector3(-direction.z, 0, direction.x).normalize();
              const controlPoint = midPoint.clone().add(perpendicular.multiplyScalar(curveOffset + 2));

              // Create quadratic bezier curve
              const curve = new THREE.QuadraticBezierCurve3(
                occ.source,
                controlPoint,
                occ.target
              );

              const points = curve.getPoints(20);
              const curveGeometry = new THREE.BufferGeometry().setFromPoints(points);
              
              // Get color based on visibility level
              const colorHex = visibilityColors[occ.visibility] || '#9CA3AF';
              const color = parseInt(colorHex.replace('#', '0x'));
              
              // Use color based on visibility
              const curveMaterial = new THREE.LineBasicMaterial({
                color: color,
                transparent: true,
                opacity: 0.6,
                linewidth: 2,
              });

              const curveLine = new THREE.Line(curveGeometry, curveMaterial);
              
              // Store metadata for interaction (for future editing support)
              const [sourceTrackId, targetTrackId] = pairKey.split("_");
              (curveLine as any).userData = {
                type: 'occlusion',
                source_track_id: sourceTrackId,
                target_track_id: targetTrackId,
                source_label: occ.sourceLabel,
                target_label: occ.targetLabel,
                visibility: occ.visibility,
                controlPoint: controlPoint.clone()
              };
              
              annotationGroup.add(curveLine);

              // Add label at midpoint with visibility level
              const visibilityText = `Visibility: ${occ.visibility}`;
              const labelCanvas = document.createElement("canvas");
              const labelContext = labelCanvas.getContext("2d");
              
              if (labelContext) {
                // Bigger canvas and font when specific target is selected
                const isTargetSelected = selectedTrackId !== "all";
                const canvasWidth = isTargetSelected ? 384 : 256;
                const canvasHeight = isTargetSelected ? 72 : 48;
                const fontSize = isTargetSelected ? 20 : 14;
                const textY = isTargetSelected ? 42 : 28;
                
                labelCanvas.width = canvasWidth;
                labelCanvas.height = canvasHeight;
                
                // Use hex color for label background
                const r = parseInt(colorHex.slice(1, 3), 16);
                const g = parseInt(colorHex.slice(3, 5), 16);
                const b = parseInt(colorHex.slice(5, 7), 16);
                labelContext.fillStyle = `rgba(${r}, ${g}, ${b}, 0.9)`;
                
                labelContext.fillRect(0, 0, canvasWidth, canvasHeight);
                labelContext.font = `Bold ${fontSize}px Arial`;
                labelContext.fillStyle = "white";
                labelContext.textAlign = "center";
                labelContext.fillText(visibilityText, canvasWidth / 2, textY);

                const labelTexture = new THREE.CanvasTexture(labelCanvas);
                const labelSpriteMaterial = new THREE.SpriteMaterial({ map: labelTexture });
                const labelSprite = new THREE.Sprite(labelSpriteMaterial);
                labelSprite.position.copy(controlPoint);
                
                // Store same metadata on sprite
                (labelSprite as any).userData = {
                  type: 'occlusion',
                  source_track_id: sourceTrackId,
                  target_track_id: targetTrackId,
                  source_label: occ.sourceLabel,
                  target_label: occ.targetLabel,
                  visibility: occ.visibility,
                  controlPoint: controlPoint.clone()
                };
                
                // Bigger sprite scale when specific target is selected
                if (isTargetSelected) {
                  labelSprite.scale.set(5, 1, 1);
                } else {
                  labelSprite.scale.set(3, 0.6, 1);
                }
                
                annotationGroup.add(labelSprite);
              }
            });
          });

          console.log(`Rendered ${occlusionPairs.size} occlusion groups`);
        }

        setLoading(false);

        // Cleanup function
        return () => {
          console.log('[LiDAR] Cleaning up Three.js resources');
          isCleanedUp = true; // Set flag to prevent further operations
          
          // Cancel animation loop first
          if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = 0;
          }
          
          // Remove event listeners
          window.removeEventListener("resize", handleResize);
          if (renderer?.domElement) {
            renderer.domElement.removeEventListener('click', handleClick);
          }
          
          // Dispose all scene objects recursively
          if (scene) {
            scene.traverse((object: any) => {
              // Dispose geometries
              if (object.geometry) {
                object.geometry.dispose();
              }
              
              // Dispose materials
              if (object.material) {
                if (Array.isArray(object.material)) {
                  object.material.forEach((material: any) => {
                    if (material.map) material.map.dispose();
                    material.dispose();
                  });
                } else {
                  if (object.material.map) object.material.map.dispose();
                  object.material.dispose();
                }
              }
              
              // Dispose textures for sprites
              if (object.isSprite && object.material?.map) {
                object.material.map.dispose();
              }
            });
            
            // Clear scene
            while (scene.children.length > 0) {
              scene.remove(scene.children[0]);
            }
          }
          
          // Dispose controls
          if (controls) {
            controls.dispose();
            controls = null;
          }
          
          // Dispose renderer and force WebGL context loss to free resources
          if (renderer) {
            renderer.dispose();
            renderer.forceContextLoss();
            renderer.context = null;
            renderer.domElement = null;
            renderer = null;
          }
          
          // Clear references
          scene = null;
          camera = null;
          annotationGroup = null;
          
          console.log('[LiDAR] Cleanup complete');
        };
      } catch (err) {
        console.error("Error loading lidar:", err);
        setError(err instanceof Error ? err.message : "Failed to load lidar");
        setLoading(false);
      }
    };

    const cleanup = initAndLoadLidar();

    return () => {
      cleanup.then((cleanupFn) => {
        if (cleanupFn) cleanupFn();
      });
    };
  }, [lidarPath, recordId, annotations, showAnnotations, spatialRelations, showRelationships, selectedTrackId, internalInteractions, showInternalInteractions, externalInteractions, showExternalInteractions, occlusions, showOcclusions]);

  return (
    <div className="relative w-full h-full bg-zinc-900 rounded-lg overflow-hidden">
      {/* Three.js Container */}
      <div ref={containerRef} className="w-full h-full" />

      {/* Stats Overlay */}
      <div className="absolute top-2 left-2 z-10 bg-black/70 text-white text-xs px-3 py-2 rounded-lg font-mono space-y-1">
        <div>Points: {stats.count.toLocaleString()}</div>
        <div>Height: {stats.minZ.toFixed(2)}m to {stats.maxZ.toFixed(2)}m</div>
        {annotations && (
          <div className="pt-2 border-t border-zinc-600">
            <div>Boxes: {annotations.bbox_3d?.length || 0}</div>
            <label className="flex items-center gap-2 cursor-pointer mt-1">
              <input
                type="checkbox"
                checked={showAnnotations}
                onChange={(e) => setShowAnnotations(e.target.checked)}
                className="w-3 h-3"
              />
              <span className="text-xs">Show Boxes</span>
            </label>
          </div>
        )}
      </div>

      {/* Loading */}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-zinc-900/80 z-20">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-4 border-zinc-600 border-t-blue-500 mx-auto mb-3"></div>
            <p className="text-white text-sm">Loading LiDAR...</p>
          </div>
        </div>
      )}

      {/* Error - show as warning banner if annotations are still visible */}
      {/* {error && !loading && (
        <div className="absolute top-2 right-2 z-20 bg-yellow-900/90 border border-yellow-600 text-yellow-200 px-4 py-3 rounded-lg max-w-md">
          <div className="flex items-start gap-2">
            <span className="text-lg">⚠️</span>
            <div className="flex-1 text-xs">
              <p className="font-semibold mb-1">Point Cloud Unavailable</p>
              <p className="opacity-90">{error}</p>
              <p className="mt-2 text-yellow-300">Showing 3D bounding boxes only</p>
            </div>
          </div>
        </div>
      )} */}

      {/* Controls Help */}
      <div className="absolute bottom-2 left-2 z-10 bg-black/70 text-white text-xs px-3 py-2 rounded-lg">
        <div>🖱️ Left: Rotate</div>
        <div>🖱️ Right: Pan</div>
        <div>🖱️ Wheel: Zoom</div>
        {selectedTrackId !== "all" && (
          <div className="mt-2 pt-2 border-t border-zinc-600">
            <div>💡 Click relationship to edit</div>
          </div>
        )}
      </div>

      {/* Relationship Editor Popup */}
      {selectedRelationship && selectedTrackId !== "all" && (
        <>
          {/* Backdrop overlay */}
          <div
            onClick={() => {
              // Auto-save when clicking outside
              if (onRelationsUpdate) {
                onRelationsUpdate([{
                  source_track_id: selectedRelationship.source_track_id,
                  target_track_id: selectedRelationship.target_track_id,
                  relations: selectedRelationship.relations,
                }]);
              }
              setSelectedRelationship(null);
            }}
            className="absolute inset-0 bg-black/40 z-[29]"
          />
          
          {/* Popup - Centered */}
          <div
            onClick={(e) => e.stopPropagation()} // Prevent backdrop click when clicking popup
            className="absolute z-30 bg-zinc-800 border-2 border-blue-500 rounded-lg shadow-2xl p-4 w-[400px] max-h-[500px] overflow-y-auto"
            style={{
              left: '50%',
              top: '50%',
              transform: 'translate(-50%, -50%)',
            }}
          >
            {/* Header */}
            <div className="flex justify-between items-start mb-3">
              <div>
                <div className="text-white font-semibold text-sm mb-1">
                  Edit Relationship
                </div>
                <div className="text-zinc-400 text-xs">
                  {selectedRelationship.source_label} → {selectedRelationship.target_label}
                </div>
                <div className="text-zinc-500 text-xs font-mono mt-1">
                  {selectedRelationship.source_track_id.substring(0, 8)}... → {selectedRelationship.target_track_id.substring(0, 8)}...
                </div>
              </div>
              <button
                onClick={() => {
                  // Auto-save when clicking X
                  if (onRelationsUpdate) {
                    onRelationsUpdate([{
                      source_track_id: selectedRelationship.source_track_id,
                      target_track_id: selectedRelationship.target_track_id,
                      relations: selectedRelationship.relations,
                    }]);
                  }
                  setSelectedRelationship(null);
                }}
                className="text-zinc-400 hover:text-white text-lg leading-none"
              >
                ✕
              </button>
            </div>

            {/* Relation toggles - Single selection */}
            <div className="space-y-2 mb-3">
              <div className="text-zinc-300 text-xs font-semibold mb-2">
                Select one relation:
              </div>
              <div className="flex flex-wrap gap-1">
                {availableRelations.map((relation) => {
                  const isSelected = selectedRelationship.relations.includes(relation);
                  return (
                    <button
                      key={relation}
                      onClick={() => {
                        // Only allow single selection
                        setSelectedRelationship({
                          ...selectedRelationship,
                          relations: [relation],
                        });
                      }}
                      className={`px-2 py-1 text-xs rounded transition-colors ${
                        isSelected
                          ? "bg-blue-500 text-white"
                          : "bg-zinc-700 text-zinc-300 hover:bg-zinc-600"
                      }`}
                    >
                      {relation.replace(/_/g, " ")}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Info text */}
            <div className="text-xs text-zinc-400 text-center mt-3 p-2 bg-blue-900/20 rounded border border-blue-700/30">
              💾 Changes are saved automatically
            </div>
          </div>
        </>
      )}

      {/* Action editor popup (Internal Interactions) */}
      {selectedInteraction && (
        <>
          {/* Backdrop overlay */}
          <div
            onClick={async () => {
              // Auto-save when clicking outside
              console.log('[Internal Int] Auto-save on backdrop click:', {
                track_id: selectedInteraction.track_id,
                actions: selectedInteraction.actions,
              });
              if (onInternalInteractionsUpdate) {
                await onInternalInteractionsUpdate([{
                  track_id: selectedInteraction.track_id,
                  actions: selectedInteraction.actions,
                }]);
              }
              setSelectedInteraction(null);
            }}
            className="absolute inset-0 bg-black/40 z-[999]"
          />
          
          {/* Popup - Centered */}
          <div
            onClick={(e) => e.stopPropagation()} // Prevent backdrop click when clicking popup
            style={{
              position: "absolute",
              left: "50%",
              top: "50%",
              transform: "translate(-50%, -50%)",
              zIndex: 1000,
            }}
            className="w-96 max-h-[500px] overflow-y-auto bg-zinc-800 border border-zinc-600 rounded-lg shadow-2xl p-4"
          >
            <div className="flex justify-between items-start mb-3">
              <div>
                <div className="text-white font-semibold text-sm mb-1">
                  Edit Actions
                </div>
                <div className="text-zinc-400 text-xs">
                  {selectedInteraction.label}
                </div>
                <div className="text-zinc-500 text-xs font-mono mt-1">
                  ID: {selectedInteraction.track_id.substring(0, 12)}...
                </div>
              </div>
              <button
                onClick={async () => {
                  // Auto-save when clicking X
                  console.log('[Internal Int] Auto-save on X click:', {
                    track_id: selectedInteraction.track_id,
                    actions: selectedInteraction.actions,
                  });
                  if (onInternalInteractionsUpdate) {
                    await onInternalInteractionsUpdate([{
                      track_id: selectedInteraction.track_id,
                      actions: selectedInteraction.actions,
                    }]);
                  }
                  setSelectedInteraction(null);
                }}
                className="text-zinc-400 hover:text-white text-lg leading-none"
              >
                ✕
              </button>
            </div>

            {/* Action toggles - Multiple selection allowed */}
            <div className="space-y-2 mb-3">
              <div className="text-zinc-300 text-xs font-semibold mb-2">
                Select actions (multiple allowed):
              </div>
              <div className="flex flex-wrap gap-1">
                {availableActions.map((action) => {
                  const isSelected = selectedInteraction.actions.includes(action);
                  const color = actionColors[action] || '#6B7280';
                  
                  return (
                    <button
                      key={action}
                      onClick={() => {
                        // Toggle action (multiple selection allowed)
                        const newActions = isSelected
                          ? selectedInteraction.actions.filter(a => a !== action)
                          : [...selectedInteraction.actions, action];
                        
                        console.log('[Internal Int] Action toggle:', {
                          action,
                          was_selected: isSelected,
                          old_actions: selectedInteraction.actions,
                          new_actions: newActions,
                        });
                        
                        setSelectedInteraction({
                          ...selectedInteraction,
                          actions: newActions,
                        });
                      }}
                      style={{
                        backgroundColor: isSelected ? color : '#3f3f46',
                        borderColor: isSelected ? color : '#52525b',
                        borderWidth: '2px',
                      }}
                      className={`px-2 py-1 text-xs rounded transition-all font-semibold ${
                        isSelected
                          ? "text-white shadow-lg"
                          : "text-zinc-300 hover:bg-zinc-600"
                      }`}
                    >
                      {action.replace(/_/g, " ")}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Clear All Button */}
            {selectedInteraction.actions.length > 0 && (
              <div className="mt-3 pt-3 border-t border-zinc-700">
                <button
                  onClick={() => {
                    console.log('[Internal Int] Clearing all actions for:', selectedInteraction.track_id);
                    setSelectedInteraction({
                      ...selectedInteraction,
                      actions: [],
                    });
                  }}
                  className="w-full px-3 py-2 bg-red-600 hover:bg-red-700 text-white text-xs font-semibold rounded transition-colors"
                >
                  🗑️ Clear All Actions
                </button>
              </div>
            )}

            {/* Info text */}
            <div className="text-xs text-zinc-400 text-center mt-3 p-2 bg-blue-900/20 rounded border border-blue-700/30">
              💾 Changes are saved automatically
            </div>
          </div>
        </>
      )}

      {/* External Interaction editor popup */}
      {selectedExternalInteraction && selectedTrackId !== "all" && (
        <>
          {/* Backdrop overlay */}
          <div
            onClick={async () => {
              // Auto-save when clicking outside
              console.log('[External Interaction] Auto-saving:', selectedExternalInteraction.interaction_type);
              if (onExternalInteractionsUpdate) {
                await onExternalInteractionsUpdate([{
                  source_track_id: selectedExternalInteraction.source_track_id,
                  target_track_id: selectedExternalInteraction.target_track_id,
                  interaction_type: selectedExternalInteraction.interaction_type,
                }]);
              }
              setSelectedExternalInteraction(null);
            }}
            className="absolute inset-0 bg-black/40 z-[29]"
          />
          
          {/* Popup - Centered */}
          <div
            onClick={(e) => e.stopPropagation()} // Prevent backdrop click when clicking popup
            className="absolute z-30 bg-zinc-800 border-2 border-green-500 rounded-lg shadow-2xl p-4 w-[400px] max-h-[500px] overflow-y-auto"
            style={{
              left: '50%',
              top: '50%',
              transform: 'translate(-50%, -50%)',
            }}
          >
            {/* Header */}
            <div className="flex justify-between items-start mb-3">
              <div>
                <div className="text-white font-semibold text-sm mb-1">
                  Edit External Interaction
                </div>
                <div className="text-zinc-400 text-xs">
                  {selectedExternalInteraction.source_label} → {selectedExternalInteraction.target_label}
                </div>
                <div className="text-zinc-500 text-xs font-mono mt-1">
                  {selectedExternalInteraction.source_track_id.substring(0, 8)}... → {selectedExternalInteraction.target_track_id.substring(0, 8)}...
                </div>
              </div>
              <button
                onClick={async () => {
                  // Auto-save when clicking X
                  console.log('[External Interaction] Saving on X click:', selectedExternalInteraction.interaction_type);
                  if (onExternalInteractionsUpdate) {
                    await onExternalInteractionsUpdate([{
                      source_track_id: selectedExternalInteraction.source_track_id,
                      target_track_id: selectedExternalInteraction.target_track_id,
                      interaction_type: selectedExternalInteraction.interaction_type,
                    }]);
                  }
                  setSelectedExternalInteraction(null);
                }}
                className="text-zinc-400 hover:text-white text-lg leading-none"
              >
                ✕
              </button>
            </div>

            {/* Interaction selection - Single selection */}
            <div className="space-y-2 mb-3">
              <div className="text-zinc-300 text-xs font-semibold mb-2">
                Select one interaction type:
              </div>
              <div className="flex flex-wrap gap-1">
                {availableExternalInteractions.map((interaction) => {
                  const isSelected = selectedExternalInteraction.interaction_type === interaction;
                  const color = externalInteractionColors[interaction] || '#6B7280';
                  
                  return (
                    <button
                      key={interaction}
                      onClick={() => {
                        // Single selection only
                        console.log('[External Interaction] Changed type to:', interaction);
                        setSelectedExternalInteraction({
                          ...selectedExternalInteraction,
                          interaction_type: interaction,
                        });
                      }}
                      style={{
                        backgroundColor: isSelected ? color : '#3f3f46',
                        borderColor: isSelected ? color : '#52525b',
                        borderWidth: '2px',
                      }}
                      className={`px-2 py-1 text-xs rounded transition-all font-semibold ${
                        isSelected
                          ? "text-white shadow-lg"
                          : "text-zinc-300 hover:bg-zinc-600"
                      }`}
                    >
                      {interaction.replace(/_/g, " ")}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Remove Interaction Button */}
            <div className="mt-3 pt-3 border-t border-zinc-700">
              <button
                onClick={async () => {
                  console.log('[External Interaction] Removing interaction:', {
                    source: selectedExternalInteraction.source_track_id,
                    target: selectedExternalInteraction.target_track_id,
                  });
                  
                  // Send update with empty interaction type to signal removal
                  // The parent component should handle removing this from the array
                  if (onExternalInteractionsUpdate) {
                    await onExternalInteractionsUpdate([{
                      source_track_id: selectedExternalInteraction.source_track_id,
                      target_track_id: selectedExternalInteraction.target_track_id,
                      interaction_type: '', // Empty = remove
                    }]);
                  }
                  setSelectedExternalInteraction(null);
                }}
                className="w-full px-3 py-2 bg-red-600 hover:bg-red-700 text-white text-xs font-semibold rounded transition-colors"
              >
                🗑️ Remove This Interaction
              </button>
            </div>

            {/* Info text */}
            <div className="text-xs text-zinc-400 text-center mt-3 p-2 bg-green-900/20 rounded border border-green-700/30">
              💾 Changes are saved automatically
            </div>
          </div>
        </>
      )}
    </div>
  );
}
