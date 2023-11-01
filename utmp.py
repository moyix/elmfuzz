# This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

import kaitaistruct
from kaitaistruct import KaitaiStruct, KaitaiStream, BytesIO


if getattr(kaitaistruct, 'API_VERSION', (0, 9)) < (0, 9):
    raise Exception("Incompatible Kaitai Struct Python API: 0.9 or later is required, but you have %s" % (kaitaistruct.__version__))

class Utmp(KaitaiStruct):
    def __init__(self, _io, _parent=None, _root=None):
        self._io = _io
        self._parent = _parent
        self._root = _root if _root else self
        self._read()

    def _read(self):
        self.records = []
        i = 0
        while not self._io.is_eof():
            self.records.append(Utmp.Record(self._io, self, self._root))
            i += 1


    class Record(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.ut_type = self._io.read_s4le()
            self.ut_pid = self._io.read_s4le()
            self.ut_line = (self._io.read_bytes(32)).decode(u"ASCII")
            self.ut_id = (self._io.read_bytes(4)).decode(u"ASCII")
            self.ut_user = (self._io.read_bytes(32)).decode(u"ASCII")
            self.ut_host = (self._io.read_bytes(256)).decode(u"ASCII")
            self.ut_exit = Utmp.Record.ExitStatus(self._io, self, self._root)
            self.ut_session = self._io.read_s4le()
            self.ut_tv = Utmp.Record.Timeval(self._io, self, self._root)
            self.ut_addr_v6 = []
            for i in range(4):
                self.ut_addr_v6.append(self._io.read_s4le())

            self.unused = []
            for i in range(20):
                self.unused.append(self._io.read_u1())


        class ExitStatus(KaitaiStruct):
            def __init__(self, _io, _parent=None, _root=None):
                self._io = _io
                self._parent = _parent
                self._root = _root if _root else self
                self._read()

            def _read(self):
                self.e_termination = self._io.read_s2le()
                self.e_exit = self._io.read_s2le()


        class Timeval(KaitaiStruct):
            def __init__(self, _io, _parent=None, _root=None):
                self._io = _io
                self._parent = _parent
                self._root = _root if _root else self
                self._read()

            def _read(self):
                self.tv_sec = self._io.read_s4le()
                self.tv_usec = self._io.read_s4le()




