# use the official Bun image
# see all versions at https://hub.docker.com/r/oven/bun/tags
FROM oven/bun:1 AS base
WORKDIR /usr/src/app

# install dependencies into temp directory
# this will cache them and speed up future builds
FROM base AS install
RUN mkdir -p /temp/dev
# COPY package.json bun.lock /temp/dev/
COPY package.json  /temp/dev/
RUN cd /temp/dev && bun install

# install with --production (exclude devDependencies)
RUN mkdir -p /temp/prod
# COPY package.json bun.lock /temp/prod/
COPY package.json  /temp/prod/
RUN cd /temp/prod && bun install --production

# copy node_modules from temp directory
# then copy all (non-ignored) project files into the image
FROM base AS prerelease
COPY --from=install /temp/dev/node_modules node_modules
COPY . .

# make sure app dir is writable by whoever runs it
RUN mkdir -p /usr/src/app/.next && chmod -R 777 /usr/src/app

EXPOSE 3000
ENTRYPOINT ["bun","dev","-H","0.0.0.0","-p","3000"]
