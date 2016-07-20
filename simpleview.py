# License: public domain
class SimpleView(object):
    """
    Simple view-like object of an array-like object.

    Slices taken with __getitem__ are combined and stored for later use, 
      but the actual data is not extracted from the underlying array-like 
      object until __array__ is called.
      
    Warnings about the current implementation: 
    - Does NOT handle slice steppings.
    - Does NOT handle numpy.newaxis or None, either.
    - Does NOT support reducing the dimensionality of the data. (e.g. a[2:3] is okay, a[2] is not).
    
    (In principle, all of the above could be fixed...)
    """
    
    def __init__( self, data, slicing=(slice(None),) ):
        self._data = data
        slicing = self._expand_slicing(slicing, data.shape)
        slicing = self._explicit_slicing(slicing, data.shape)
        self._slicing = slicing

    def __array__(self):
        return self._data[self._slicing]
    
    @property
    def shape(self):
        return tuple(s.stop - s.start for s in self._slicing)
    
    @property
    def dtype(self):
        return self._data.dtype
    
    def __getitem__(self, slicing):
        assert all( isinstance(s, slice) for s in slicing ), "Sorry, SimpleView doesn't support integer slicing args yet."
        slicing = self._expand_slicing(slicing, self.shape)
        slicing = self._explicit_slicing(slicing, self.shape)
        combined_slicing = self._combine_slicings(self._slicing, slicing)
        return SimpleView(self._data, combined_slicing)
    
    def __setitem__(self, slicing, new_data):
        slicing = self._expand_slicing(slicing, self.shape)
        slicing = self._explicit_slicing(slicing, self.shape)
        combined_slicing = self._combine_slicings(self._slicing, slicing)
        self._data[combined_slicing] = new_data
    
    @classmethod
    def _combine_slicings(cls, slicing1, slicing2):
        assert len(slicing1) == len(slicing2)
        final_slicing = ()
        for s1, s2 in zip(slicing1, slicing2):
            # This code needs to be a little more complicated if we want to support steps other than 1.
            # For now, we just disallow that case.
            assert s1.step == None or s1.step == 1, "SimpleView does not support slice step sizes other than 1"
            assert s2.step == None or s2.step == 1, "SimpleView does not support slice step sizes other than 1"
            start = s1.start + s2.start
            stop = s1.start + s2.stop
            final_slicing += (slice(start, stop),)
        return final_slicing
    
    @classmethod
    def _explicit_slicing(cls, slicing, shape):
        """
        Replace all slice(None) items in the given slicing with 
          explicit start/stop coordinates using the given shape.
        Also, replace negative indexes with positive equivalents.
        """
        explicit_slicing = ()
        for slc, maxstop in zip(slicing, shape):
            if not isinstance(slc, slice):
                explicit_slicing += (slc,)
            else:
                start, stop, step = slc.start, slc.stop, slc.step
                if start is None:
                    start = 0
                if stop is None:
                    stop = maxstop
                if start < 0:
                    start = maxstop + start
                if stop < 0:
                    stop = maxstop + stop
                explicit_slicing += ( slice(start, stop, step), )
        return explicit_slicing

    @classmethod
    def _expand_slicing(cls, s, shape):
        """
        Args:
            s: Anything that can be used as a numpy array index:
               - int
               - slice
               - Ellipsis (i.e. ...)
               - Some combo of the above as a tuple or list
            
            shape: The shape of the array that will be accessed
            
        NOTE: Does not handle numpy.newaxis or None
            
        Returns:
            A tuple of length N where N=len(shape)
            slice(None) is inserted in missing positions so as not to change the meaning of the slicing.
            e.g. if shape=(1,2,3,4,5):
                0 --> (0,:,:,:,:)
                (0:1) --> (0:1,:,:,:,:)
                : --> (:,:,:,:,:)
                ... --> (:,:,:,:,:)
                (0,0,...,4) --> (0,0,:,:,4)            
        """
        if type(s) == list:
            s = tuple(s)
        if type(s) != tuple:
            # Convert : to (:,), or 5 to (5,)
            s = (s,)
    
        # Compute number of axes missing from the slicing
        if len(shape) - len(s) < 0:
            assert s == (Ellipsis,) or s == (slice(None),), \
                "Slicing must not have more elements than the shape, except for [:] and [...] slices"
    
        # Replace Ellipsis with (:,:,:)
        if Ellipsis in s:
            ei = s.index(Ellipsis)
            s = s[0:ei] + (len(shape) - len(s) + 1)*(slice(None),) + s[ei+1:]
    
        # Shouldn't be more than one Ellipsis
        assert Ellipsis not in s, \
            "illegal slicing: found more than one Ellipsis"

        # Append (:,) until we get the right length
        s += (len(shape) - len(s))*(slice(None),)
        
        # Special case: we allow [:] and [...] for empty shapes ()
        if shape == ():
            s = ()
        
        return s

# Quick test
if __name__ == "__main__":
    import numpy
    a = numpy.random.random((100,100,100)).astype(numpy.float32)
    
    sv1 = SimpleView(a)
    (sv1[10:90, 20:80, 30:70][20:40, 40:60, :-1] == a[10:90, 20:80, 30:70][20:40, 40:60, :-1]).all()

    import h5py
    f = h5py.File('foo.h5', driver='core', backing_store=False, mode='w')
    f['dset'] = a
        
    sv2 = SimpleView(f['dset'])
    (sv2[10:90, 20:80, 30:70][20:40, 40:60, :-1] == a[10:90, 20:80, 30:70][20:40, 40:60, :-1]).all()

    # Try writing
    sv2[10:90, 20:80, 30:70][20:40, 40:60, :-1] = 1.0
    assert (f['dset'][:] == sv2).all()
    
    a[10:90, 20:80, 30:70][20:40, 40:60, :-1] = 1.0
    assert (f['dset'][:] == a).all()

    #print numpy.sum(sv1)