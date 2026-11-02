# DriveSpatial

This directory contains the source code for the **DriveSpatial** project, a platform for spatial understanding and question answering in autonomous driving scenarios.

## Directory Overview

```
DriveSpatial-private/
├── drivespatial-api/
├── drivespatial-ui/
└── drivespatial-qa/
```

### `drivespatial-api/`
Backend API service that handles data processing and serves requests from the frontend. This includes:
- REST API endpoints
- Data fetching and processing
- Integration with external services

### `drivespatial-ui/`
Frontend web application built with Next.js. This includes:
- User interface components
- Evaluation and question visualization pages
- API route handlers for HPC integration

### `drivespatial-qa/`
Question answering and data collection pipeline. This includes:
- Data collection algorithms and utilities
- Question generation tasks and samplers
- Processing utilities