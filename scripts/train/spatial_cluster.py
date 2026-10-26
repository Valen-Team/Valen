"""Repository CLI for the packaged spatial replica coordinator."""
from valen.training.spatial_cluster import SpatialCluster, main

__all__ = ["SpatialCluster", "main"]


if __name__ == "__main__":
    main()
