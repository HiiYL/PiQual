import flickrapi


api_key = u'812d64b9b2dabbeba29885ca83417327'
api_secret = u'61e583b853d9ede1'

flickr = flickrapi.FlickrAPI(api_key, api_secret, format='parsed-json')
sets   = flickr.photosets.getList(user_id='73509078@N00')
title  = sets['photosets']['photoset'][0]['title']['_content']

print('First set title: %s' % title)