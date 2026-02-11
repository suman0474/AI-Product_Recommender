# ============================================================================
# EnGenie Frontend - Dockerfile
# ============================================================================
# Multi-stage build: Node for building, Nginx for serving

# Stage 1: Build the React application
FROM node:20-alpine AS builder

# Set working directory
WORKDIR /app

# Install dependencies first (for better caching)
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci --legacy-peer-deps

# Copy source code
COPY . .

# Build arguments for environment configuration
ARG VITE_API_URL
ARG VITE_ENV=production

# Set environment variables for build
ENV VITE_API_URL=${VITE_API_URL:-/api} \
    VITE_ENV=${VITE_ENV}

# Build the application
RUN npm run build

# ============================================================================
# Stage 2: Production - Nginx to serve static files
# ============================================================================
FROM nginx:alpine AS production

# Labels
LABEL maintainer="EnGenie Team" \
    version="1.0.0" \
    description="EnGenie Frontend - React/Vite Application"

# Remove default nginx config
RUN rm /etc/nginx/conf.d/default.conf

# Copy custom nginx configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Create cache directories with proper permissions
RUN mkdir -p /var/cache/nginx && \
    chown -R nginx:nginx /var/cache/nginx && \
    chown -R nginx:nginx /usr/share/nginx/html && \
    chmod -R 755 /usr/share/nginx/html

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost:80/health || exit 1

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
