import { NextRequest, NextResponse } from 'next/server';

/**
 * API Route to proxy HPC image requests with authentication
 * This handles the authentication server-side to keep credentials secure
 */
export async function GET(request: NextRequest): Promise<NextResponse> {
  try {
    const searchParams = request.nextUrl.searchParams;
    const imagePath = searchParams.get('path');

    if (!imagePath) {
      return NextResponse.json(
        { error: 'Image path is required' },
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

    const baseUrlWithSlash = hpcBaseUrl.endsWith('/') ? hpcBaseUrl : `${hpcBaseUrl}/`;
    const fullUrl = `${baseUrlWithSlash}${imagePath}`;
    
    console.log(`Fetching image from HPC: ${fullUrl}`);

    // Create Basic Auth header
    const authHeader = `Basic ${Buffer.from(`${hpcUsername}:${hpcPassword}`).toString('base64')}`;

    // Fetch the image from HPC
    const response = await fetch(fullUrl, {
      method: 'GET',
      headers: {
        'Authorization': authHeader,
      },
    });

    if (!response.ok) {
      console.error(`HPC request failed: ${response.status} ${response.statusText}`);
      return NextResponse.json(
        { error: `Failed to fetch image: ${response.statusText}` },
        { status: response.status }
      );
    }

    // Get the image data
    const imageBuffer = await response.arrayBuffer();
    const contentType = response.headers.get('content-type') || 'image/jpeg';

    // Return the image with appropriate headers
    return new NextResponse(imageBuffer, {
      status: 200,
      headers: {
        'Content-Type': contentType,
        'Cache-Control': 'public, max-age=3600',
      },
    });

  } catch (error) {
    console.error('Error fetching image from HPC:', error);
    return NextResponse.json(
      { 
        error: 'Internal server error',
        details: error instanceof Error ? error.message : 'Unknown error'
      },
      { status: 500 }
    );
  }
}
