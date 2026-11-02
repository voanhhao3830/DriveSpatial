import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  devIndicators: false,
  turbopack: {
    // Empty turbopack config to silence warning
  },
  transpilePackages: ["three", "@react-three/fiber", "@react-three/drei"],
  allowedDevOrigins: [
    "localhost:8080",
    "127.0.0.1:8080",
    "*",
    "drivespatial-ui:3000",
    "drivespatial-api:8000",

  ],
};

export default nextConfig;
