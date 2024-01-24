import pytest
import os
import cv2
import numpy as np

# Test if cv2 is working properly
def test_cv2():
    assert cv2.__version__ == "4.9.0"
