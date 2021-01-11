/* Create version-related dynamic output: instert_version_links() will output
   <li> elements for each string in `version`. The content will be an <a>,
   with the href to the parallel document in the other version. Using jquery
   ajax, it will check that the document exists, if not the href will point to
   the root of the other version
 */

// These are all the valid directories in pytorch/pytorch.github.io/docs
var versions = ['master',  '0.7.0',
               ];

function insert_version_links() {
    for (i = 0; i < versions.length; i++){
        open_list = '<li>'
        label = 'v' + versions[i];
        switch (label){
            case 'vmaster':
                label = 'master (unstable)';
                break;
            case '0.7.0':
                label += ' (stable release)';
                break;
        }
        if (typeof(DOCUMENTATION_OPTIONS) !== 'undefined') {
            if (DOCUMENTATION_OPTIONS['VERSION'] == versions[i])
            {
                open_list = '<li id="current">'
            }
        }
        /* CI will be       https://circle-artifacts.com/0/docs/distributed.html
           deployed will be https://pytorch.org/audio/stable/index.html
                         or https://pytorch.org/audio/1.7.0/index.html
                   or maybe https://pytorch.org/audio/1.7.1rc2/index.html
           local build will be file:///tmp/pytorch/audio/build/html/index.html
           So we want to capture '/audio', then any of '/build/html',
           '/' + version[i], or an empty string (the final '|'), in that order.
         */ 
        const pathPattern = /audio(\/build\/html\/|\/stable|\/master|\/[0-9.rc]+|)(.*)/;
        const m = location.pathname.match(pathPattern);
        base_url = 'https://pytorch.org/audio/' + versions[i];
        if (m == null) {
            url = base_url
        } else {
            url = 'https://pytorch.org/audio/' + versions[i] + m[2];
            // Checks synchronously if the page exists, fail -> replace the link
            $.ajax({url: url, cache: true, async: false}).fail(function(jqXHR, textStatus) {
                url = base_url
                });
        }
        document.write(open_list);
        document.write('<a id=ID href="URL">VERSION</a> </li>\n'
                        .replace('VERSION', label)
                        .replace('URL', url));
    }
}

