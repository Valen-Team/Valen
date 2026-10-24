FROM docker.io/vllm/vllm-openai:nightly

ENV PYTHONUNBUFFERED=1 VLLM_PLUGINS=valen VLLM_USE_V2_MODEL_RUNNER=0
COPY serving/vllm /opt/valen/plugin
RUN python3 -m pip install /opt/valen/plugin

WORKDIR /opt/valen
EXPOSE 8000
ENTRYPOINT ["python3", "-m", "valen_vllm.launch_server", "--host", "0.0.0.0"]
CMD []
