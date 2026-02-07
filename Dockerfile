FROM ubuntu:24.04

# Redeclare ARGs after FROM to make them available in build stages
ARG UID=1000
ARG GID=1000
ARG USERNAME=appuser

# Combine RUN commands to reduce layers and use --no-install-recommends
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create user with specified UID/GID
RUN groupadd -g ${GID} ${USERNAME} && \
    useradd -m -u ${UID} -g ${GID} -s /bin/bash ${USERNAME}

# Set working directory
WORKDIR /app

# Copy and install requirements (these change less frequently)
COPY --chown=${USERNAME}:${USERNAME} requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt --break-system-packages

# Switch to non-root user
USER ${USERNAME}