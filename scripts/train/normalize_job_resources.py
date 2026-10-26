"""Keep an existing job's CPU/RAM request equal to its submitted configuration."""
import argparse
import copy
import json


def normalize(job_id, gpus, cpus, memory_mib, namespace=None, replicas=1):
    from brainpp.rjob.client import RJobClient
    client = RJobClient(namespace=namespace)
    job = client.api.get_namespaced_custom_object(client.group, client.version, client.namespace,
                                                  client.plural, job_id, _request_timeout=30)
    tasks = job["spec"]["taskSpecs"]
    if len(tasks) != 1:
        raise ValueError("Resource normalization requires a single-task job")
    name, task = next(iter(tasks.items()))
    containers = copy.deepcopy(task["template"]["spec"]["containers"])
    if len(containers) != 1 or task["replicas"] != replicas:
        raise ValueError("Resource normalization requires one container and the expected replica count")
    resources = containers[0]["resources"]
    if any(int(resources[kind].get("nvidia.com/gpu", 0)) != gpus for kind in ("limits", "requests")):
        raise ValueError("The scheduler changed the GPU count; inspect before proceeding")
    expected = {"cpu": str(cpus), "memory": f"{memory_mib}Mi", "nvidia.com/gpu": str(gpus)}
    for kind in ("limits", "requests"):
        resources[kind].update(expected)
    client.patch(job_id, {"spec": {"taskSpecs": {name: {"template": {"spec": {"containers": containers}}}}}})
    print(json.dumps({"job_id": job_id, "requested": expected}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--cpus", type=int, required=True)
    parser.add_argument("--memory-mib", type=int, required=True)
    parser.add_argument("--namespace")
    parser.add_argument("--replicas", type=int, default=1)
    args = parser.parse_args()
    normalize(args.job_id, args.gpus, args.cpus, args.memory_mib, args.namespace, args.replicas)


if __name__ == "__main__":
    main()
