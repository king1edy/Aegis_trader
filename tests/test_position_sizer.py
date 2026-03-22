from my_src.position_sizer import PositionSizer

def test_calculate_lot_size():
    """
    Test that PositionSizer.calculate_lot_size() returns correct lot size.
    """
    sizer = PositionSizer()
    result = sizer.calculate_lot_size(balance=1000, risk_percent=2, stop_loss_pips=50, pip_value=10)
    expected = round((1000 * 0.02) / (50 * 10), 2)  # should equal 0.04
    assert result == expected