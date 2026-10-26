"""Evaluate RefCOCO-family and PhraseCut masks at original image resolution."""
from evaluation.spatial import main

if __name__ == "__main__":
    main("seg")
