FROM node:20-slim
WORKDIR /app

# copy npm metadata and the tarball
RUN echo "Copying npm metadata and tarball"
RUN ls -la
COPY package.json package-lock.json* ./
COPY wllama-wllama-2.3.5.tgz ./

RUN echo "Copied npm metadata and tarball"

RUN ls -la

RUN npm install --no-audit --prefer-offline

# copy rest of source
COPY . .
RUN echo "Copied rest of source"
RUN ls -la

# RUN cat .env

RUN npm run build

EXPOSE 3000
CMD ["npm", "run", "start"]
