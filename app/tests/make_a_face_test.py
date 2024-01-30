import pytest
from unittest.mock import Mock, patch
import cv2
import numpy as np
from ..make_a_face_module import MakeAFace

# Fixture to create a mock for cv2.VideoCapture
@pytest.fixture
def mock_video_capture():
    with patch('cv2.VideoCapture', autospec=True) as mock_cap:
        yield mock_cap

# Fixture to create a mock for PySimpleGUI window
@pytest.fixture
def mock_pysimplegui_window():
    with patch('PySimpleGUI.Window', autospec=True) as mock_window:
        mock_window.read.return_value = ()
        yield mock_window

# handle start test
@patch.object(MakeAFace, "load_and_init_current_model")
def test_handle_start(mock_pysimplegui_window):
    app = MakeAFace()
    app.window = mock_pysimplegui_window
    values = {'-PLAYER_NAME-': 'TestPlayer'}
    app.handle_start(values)

    assert app.pause_active is False
    assert app.window['-PAUSE-'].Update.called
    assert app.window['-MODELS-'].Update.called
    assert app.window['-CURRENT_MODEL-'].Update.called
    assert app.window['-NO_AHEAGO-'].Update.called
    assert app.window['-SCORE-'].Update.called
    assert app.window['-HIGHSCORES-'].Update.called

# classify face test non CI/CD  compatible
# @patch.object(MakeAFace, "lite_model", Mock(return_value=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])))
# def test_classify_face(mock_video_capture):
#     app = MakeAFace()
#     app.video_cap = mock_video_capture
#     # Set up a mock frame
#     mock_frame = np.zeros((224, 224, 3), dtype=np.uint8)
#     mock_video_capture.read.return_value = (True, mock_frame)

#     # app.lite_model = Mock(return_value=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
#     result = app.classify_face(mock_frame)

#     assert app.lite_model.called
#     # assert app.interpreter.set_tensor.called
#     # assert app.interpreter.invoke.called
#     # assert app.interpreter.get_tensor.called
#     assert result == 1 
