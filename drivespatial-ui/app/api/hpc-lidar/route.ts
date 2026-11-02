import { NextRequest, NextResponse } from 'next/server';

/**
 * API Route to proxy HPC lidar file requests with authentication
 * Handles binary lidar data (.bin, .pcd files)
 */
export async function GET(request: NextRequest): Promise<NextResponse> {
  try {
    const searchParams = request.nextUrl.searchParams;
    const lidarPath = searchParams.get('path');

    if (!lidarPath) {
      return NextResponse.json(
        { error: 'Lidar path is required' },
        { status: 400 }
      );
    }

    // Get credentials from environment variables
    const hpcUsername = process.env.HPC_USERNAME;
    const hpcPassword = process.env.HPC_PASSWORD;
    const hpcBaseUrl = process.env.HPC_BASE_URL;

    if (!hpcUsername || !hpcPassword) {
      console.error('HPC credentials not configured');
      return NextResponse.json(
        { error: 'HPC credentials not configured' },
        { status: 500 }
      );
    }

    // Construct the full URL
    const baseUrlWithSlash = hpcBaseUrl.endsWith('/') ? hpcBaseUrl : `${hpcBaseUrl}/`;
    const fullUrl = `${baseUrlWithSlash}${lidarPath}`;
    
    console.log(`Fetching lidar from HPC: ${fullUrl}`);

    // Create Basic Auth header
    const authHeader = `Basic ${Buffer.from(`${hpcUsername}:${hpcPassword}`).toString('base64')}`;

    // Fetch the lidar file from HPC
    const response = await fetch(fullUrl, {
      method: 'GET',
      headers: {
        'Authorization': authHeader,
      },
    });

    if (!response.ok) {
      console.error(`HPC request failed: ${response.status} ${response.statusText}`);
      return NextResponse.json(
        { error: `Failed to fetch lidar: ${response.statusText}` },
        { status: response.status }
      );
    }

    // Get the lidar data as binary
    const lidarBuffer = await response.arrayBuffer();
    
    // Determine content type based on file extension
    const contentType = lidarPath.endsWith('.pcd') 
      ? 'application/octet-stream' 
      : 'application/octet-stream';

    // Return the lidar data with appropriate headers
    return new NextResponse(lidarBuffer, {
      status: 200,
      headers: {
        'Content-Type': contentType,
        'Cache-Control': 'public, max-age=3600',
        'Access-Control-Allow-Origin': '*',
      },
    });

  } catch (error) {
    console.error('Error fetching lidar from HPC:', error);
    return NextResponse.json(
      { 
        error: 'Internal server error',
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}
