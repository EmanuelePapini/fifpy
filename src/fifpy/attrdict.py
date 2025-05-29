"""
This module provides a dictionary subclass that allows attribute-style access to dictionary keys.
It is directly copied from the `attrd` pypi repository, distributed under the GPL license.
The original repository can be found at:
https://pypi.org/project/attrd/
The original license is as follows:
Copyright (C) 2016-2020  David F. McCarthy
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,     
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

class AttrDict(dict):
    def __process_args(self, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, dict):
                    obj[k] = AttrDict(v)
                elif isinstance(v, (tuple, list)):
                    obj[k] = self.__process_args(v)
        elif isinstance(obj, (tuple, list)):
            for i, v in enumerate(obj):
                if isinstance(v, dict):
                    obj[i] = AttrDict(v)

        return obj

    def __init__(self, obj=None, **kwargs):
        obj = self.__process_args(obj)
        kwargs = self.__process_args(kwargs)
        super().__init__(obj or {}, **kwargs)

    def __getattr__(self, item: str):
        if item in self:
            return self[item]
        else:
            return super().__getattribute__(item)

    def __getattribute__(self, item: str):
        if item != 'items' and item in self:
            return self[item]
        else:
            return super().__getattribute__(item)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, item):
        del self[item]

    def __getstate__(self):
        return dict(self)

    @staticmethod
    def __setstate__(obj):
        return obj

AttrDictSens = AttrDict