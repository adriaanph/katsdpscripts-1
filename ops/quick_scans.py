# The *with* keyword is standard in Python 2.6, but has to be explicitly imported in Python 2.5
from __future__ import with_statement

import katpoint
import katuilib
from katuilib import CaptureSession
import uuid

kat = katuilib.tbuild('cfg-local.ini', 'local_ff')

cat = katpoint.Catalogue(file('/var/kat/conf/source_list.csv'), add_specials=False, antenna=kat.sources.antenna)
cat.remove('Zenith')
cat.add('Jupiter, special')

with CaptureSession(kat, str(uuid.uuid1()), 'ffuser', 'Quick scan example', kat.ants) as session:

    for target in cat.iterfilter(el_limit_deg=5):
        session.raster_scan(target)
