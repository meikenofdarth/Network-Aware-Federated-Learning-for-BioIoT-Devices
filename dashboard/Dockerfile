# ── Stage 1: build the React app ─────────────────────────────────────
FROM node:20-slim AS builder
WORKDIR /app

# Install dependencies first (layer cache)
COPY dashboard/package.json .
RUN npm install

# Copy source and build
COPY dashboard/ .
RUN npm run build

# ── Stage 2: serve with nginx ─────────────────────────────────────────
FROM nginx:alpine AS runner

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Custom nginx config:
# - Serves static files from /usr/share/nginx/html
# - Proxies /api/* to the dashboard-api service
# - Handles SPA routing (all unknown routes → index.html)
COPY dashboard/nginx.conf /etc/nginx/conf.d/default.conf

COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]