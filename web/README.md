# Smart Traffic Route Planner — Frontend

This small Next.js app calls the backend API to request routes and visualize the graph.

Local dev
1. cd web
2. cp .env.local.example .env.local
3. Set NEXT_PUBLIC_API_URL in .env.local (default: http://localhost:8000)
4. npm install
5. npm run dev

Deploy
- Deploy this folder to Vercel and set NEXT_PUBLIC_API_URL to your backend URL.
