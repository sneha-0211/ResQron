FROM node:18
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci || npm install

# Copy sources
COPY . .

EXPOSE 5173
CMD ["npm","run","dev","--","--host"]
