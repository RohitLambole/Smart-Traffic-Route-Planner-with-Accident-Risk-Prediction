# Deploying to Render (Docker)

This project now uses real OpenStreetMap data (osmnx) as the single graph source. The repository
includes a Dockerfile that installs the required system libraries and Python dependencies.

Quick steps to deploy to Render using the included Dockerfile:

1. Push your code to the `dev` branch on GitHub (the Dockerfile is in the repo root).
2. In Render, create a new Service -> Web Service.
3. Connect your GitHub repo and select the `dev` branch.
4. Choose 'Docker' as the Environment (Render will build the provided Dockerfile).
5. Set environment variables (in Render's Dashboard):
   - OSM_PLACE = "Pune, India"   # or your desired place name
   - PORT = 8000
6. Deploy. Render will build the image and run the container. The app will build the OSM graph for the configured place.

Local development notes (you can run locally using Docker):

# Build the image
docker build -t smart-traffic-api .

# Run the container (example)
docker run -p 8000:8000 -e OSM_PLACE="Pune, India" smart-traffic-api

The API will be available at http://localhost:8000
