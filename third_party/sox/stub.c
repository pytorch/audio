#include <sox.h>

int sox_add_effect(
    sox_effects_chain_t* chain,
    sox_effect_t* effp,
    sox_signalinfo_t* in,
    sox_signalinfo_t const* out) {
  return -1;
}
int sox_close(sox_format_t* ft) {
  return -1;
}

sox_effect_t* sox_create_effect(sox_effect_handler_t const* eh) {
  return NULL;
}

sox_effects_chain_t* sox_create_effects_chain(
    sox_encodinginfo_t const* in_enc,
    sox_encodinginfo_t const* out_enc) {
  return NULL;
}

void sox_delete_effect(sox_effect_t* effp) {}
void sox_delete_effects_chain(sox_effects_chain_t* ecp) {}

int sox_effect_options(sox_effect_t* effp, int argc, char* const argv[]) {
  return -1;
}

const sox_effect_handler_t* sox_find_effect(char const* name) {
  return NULL;
}

int sox_flow_effects(
    sox_effects_chain_t* chain,
    int callback(sox_bool all_done, void* client_data),
    void* client_data) {
  return -1;
}

const sox_effect_fn_t* sox_get_effect_fns(void) {
  return NULL;
}

const sox_format_tab_t* sox_get_format_fns(void) {
  return NULL;
}

sox_globals_t* sox_get_globals(void) {
  return NULL;
}

sox_format_t* sox_open_read(
    char const* path,
    sox_signalinfo_t const* signal,
    sox_encodinginfo_t const* encoding,
    char const* filetype) {
  return NULL;
}

sox_format_t* sox_open_write(
    char const* path,
    sox_signalinfo_t const* signal,
    sox_encodinginfo_t const* encoding,
    char const* filetype,
    sox_oob_t const* oob,
    sox_bool overwrite_permitted(char const* filename)) {
  return NULL;
}

const char* sox_strerror(int sox_errno) {
  return NULL;
}

size_t sox_write(sox_format_t* ft, const sox_sample_t* buf, size_t len) {
  return 0;
}

int sox_init() {
  return -1;
};
int sox_quit() {
  return -1;
};
