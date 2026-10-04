# ---------- Stage 1: build ----------
FROM node:20-alpine AS build

WORKDIR /app

# Install deps first (layer caching)
COPY package.json package-lock.json ./
RUN npm ci

# Copy source
COPY . .

# Vite inlines VITE_* at build time
ARG VITE_API_URL=/api
ARG VITE_MLFLOW_URL=http://localhost:5000
ENV VITE_API_URL=$VITE_API_URL
ENV VITE_MLFLOW_URL=$VITE_MLFLOW_URL

RUN npm run build

# ---------- Stage 2: serve ----------
FROM nginx:1.27-alpine

# Custom nginx config
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built assets
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]