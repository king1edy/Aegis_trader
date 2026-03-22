from src.message_formatter import MessageFormatter

def test_format_trade():
    formatter = MessageFormatter()
    result = formatter.format_trade("BUY", 1900)
    assert result == "BUY GOLD at 1900"