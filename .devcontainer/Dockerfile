FROM ghcr.io/converged-computing/google-performance-study:kripke-caliper-test
# FROM fluxrm/flux-sched:jammy

LABEL maintainer="Vanessasaurus <@vsoch>"

# Match the default user id for a single system so we aren't root
ARG USER_UID=1000
ARG USER_GID=1000
ENV USERNAME=${USERNAME}
ENV USER_UID=${USER_UID}
ENV USER_GID=${USER_GID}
RUN apt-get update && apt-get install -y less python3-pip hwloc
