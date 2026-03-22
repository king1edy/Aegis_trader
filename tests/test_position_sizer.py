from src.position_sizer import PositionSizer

def test_calculate_lot_size():
    sizer = PositionSizer()
    result = sizer.calculate_lot_size(1000,2)
    assert result == 20