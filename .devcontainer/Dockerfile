ARG VARIANT=ubuntu-22.04
FROM mcr.microsoft.com/vscode/devcontainers/base:0-${VARIANT}

# Install dependencies
RUN apt-get update && \
    apt-get install -y curl libicu-dev libxml2-dev libsqlite3-dev && \
    rm -rf /var/lib/apt/lists/*

# Download and install Swift
RUN curl -s https://archive.swiftlang.xyz/install.sh | bash && \
    apt-get install -y swiftlang
