/* Create version-related dynamic output: find_parallel_page() will output
   <li> elements for each string in `version`. The content will be an <a>,
   with the href to the parallel document in the other version. It will check
   that the document exists, if not the href will point to the root of the
   other version
 */

// These are all the valid directories in pytorch/pytorch.github.io/docs
var versions = ['master',  '0.7.0',
               ];

// Set up asynchronous calls to populate the version pulldown.
function UrlExists(id, label, url_yes, url_no, callback)
{
    var http = new XMLHttpRequest();
    http.open('HEAD', url_yes);
    http.onreadystatechange = function() {
        if (this.readyState == this.DONE) {
            if (this.status == 404) {
                callback(id, label, url_no);
            } else {
                callback(id, label, url_yes);
            }
        }
    };
    http.send();
}
// A callback that will add a new value to the list. The link will be either
// - the page in an different version (if available)
// - the search page in a different version (if not available)
function insert_version_links(id, label, url) {
    document.getElementById(id).innerHTML='<a id=ID href="URL">VERSION</a> </li>\n'
                    .replace('VERSION', label)
                    .replace('URL', url);
}

// Now the wrapper for the callback above. It will use a UrlExists to try
// to find the page in the other version
function find_parallel_page() {
    for (i = 0; i < versions.length; i++){
        id = 'v' + versions[i];
        label = id;
        switch (label){
            case 'vmaster':
                label = 'master (unstable)';
                break;
            case 'v0.7.0':
                label += ' (stable release)';
                break;
        }
        /* We want the page's name, which could be a few directories deep.
           For the page generated/torch.Generator.html
           CI will be       https://circle-artifacts.com/0/docs/generated/torch.Generator.html
           deployed will be https://pytorch.org/audio/stable/generated/torch.Generator.html
                   or maybe https://pytorch.org/audio/0.8.0rc2/generated/torch.Generator.html
           local build will be file:///audio/docs/build/html/generated/torch.Generator.html
           So we want to capture '/audio', then any of '/build/html',
           '/' + version[i], or an empty string (the final '|'), in that order.
         */
        const pathPattern = /audio(\/docs\/build\/html|\/stable|\/master|\/[0-9.rc]+|)(.*)/;
        const m = location.pathname.match(pathPattern);
        base_url = 'https://pytorch.org/audio/' + versions[i];
        if (m == null) {
            url_yes = base_url
        } else {
            url_yes = 'https://pytorch.org/audio/' + versions[i] + m[2];
        }
        element = '<li id=ID></li>'.replace('ID', id)
        if (typeof(DOCUMENTATION_OPTIONS) !== 'undefined') {
            if (DOCUMENTATION_OPTIONS['VERSION'] == versions[i])
            {
                element = '<li id=ID class="current"></li>'.replace('ID', id)
            }
        }
        document.write(element)
        UrlExists(id, label, url_yes, base_url, insert_version_links);
    }
}

