# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: allen_blink.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)




DESCRIPTOR = _descriptor.FileDescriptor(
  name='allen_blink.proto',
  package='',
  serialized_pb='\n\x11\x61llen_blink.proto\";\n\x05\x42link\x12\x0c\n\x04\x65\x61rs\x18\x01 \x03(\x01\x12\x11\n\tblink_idx\x18\x02 \x03(\x05\x12\x11\n\tblink_num\x18\x03 \x01(\x05\"5\n\x03Yaw\x12\x0c\n\x04\x65\x61rs\x18\x01 \x03(\x01\x12\x0f\n\x07yaw_idx\x18\x02 \x03(\x05\x12\x0f\n\x07yaw_num\x18\x03 \x01(\x05')




_BLINK = _descriptor.Descriptor(
  name='Blink',
  full_name='Blink',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='ears', full_name='Blink.ears', index=0,
      number=1, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='blink_idx', full_name='Blink.blink_idx', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='blink_num', full_name='Blink.blink_num', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=21,
  serialized_end=80,
)


_YAW = _descriptor.Descriptor(
  name='Yaw',
  full_name='Yaw',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='ears', full_name='Yaw.ears', index=0,
      number=1, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='yaw_idx', full_name='Yaw.yaw_idx', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='yaw_num', full_name='Yaw.yaw_num', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  serialized_start=82,
  serialized_end=135,
)

DESCRIPTOR.message_types_by_name['Blink'] = _BLINK
DESCRIPTOR.message_types_by_name['Yaw'] = _YAW

class Blink(_message.Message):
  __metaclass__ = _reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _BLINK

  # @@protoc_insertion_point(class_scope:Blink)

class Yaw(_message.Message):
  __metaclass__ = _reflection.GeneratedProtocolMessageType
  DESCRIPTOR = _YAW

  # @@protoc_insertion_point(class_scope:Yaw)


# @@protoc_insertion_point(module_scope)
