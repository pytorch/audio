
#include <TH/TH.h>

#include <sox.h>

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch., Real, Tensor)
#define libthsox_(NAME) TH_CONCAT_4(libthsox_, Real, _, NAME)

#include "generic/th_sox.c"
#include "THGenerateAllTypes.h"

