from my_src.message_formatter import MessageFormatter

def test_format_trade():
    formatter = MessageFormatter()
    result = formatter.format_trade("BUY", 1900)
    expected = "Trade: BUY at 1900"
    assert result == expected