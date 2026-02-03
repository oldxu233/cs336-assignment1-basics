# null_char = chr(0)
# print("this is a test" + null_char + "string")
# repr_result = repr(null_char)
# print("this is a test" + repr_result + "string")

# test_string = "hello! こんにちは!"
# utf8_encoded = test_string.encode("utf-8")
# print("utf8_encoded: ", utf8_encoded)
# print("list: ", list(utf8_encoded))
# print(len(test_string))
# print(len(utf8_encoded))
# print(utf8_encoded.decode("utf-8"))

# def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
#     return "".join([bytes([b]).decode("utf-8") for b in bytestring])
# print(decode_utf8_bytes_to_str_wrong("hello".encode("utf-8")))
# print(f"错误解码：{decode_utf8_bytes_to_str_wrong("你".encode("utf-8"))}")

# invalid2 = bytes([0b11000001, 0b10000001])  # 0xC1 0x81
# try:
#     invalid2.decode('utf-8')
# except UnicodeDecodeError as e:
#     print(f"错误: {e}")

