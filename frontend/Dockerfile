ARG NODE_BASE_IMAGE=node:20-alpine
ARG NGINX_BASE_IMAGE=nginx:1.27-alpine

FROM ${NODE_BASE_IMAGE} AS build
WORKDIR /app

ARG NPM_REGISTRY=

COPY package.json /app/package.json
RUN if [ -n "${NPM_REGISTRY}" ]; then npm config set registry "${NPM_REGISTRY}"; fi \
  && npm install

COPY tsconfig.json vite.config.ts index.html /app/
COPY src /app/src

RUN npm run build

FROM ${NGINX_BASE_IMAGE}
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
