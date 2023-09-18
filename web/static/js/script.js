function BtnLoading(elem) {
  $(elem).attr("data-original-text", $(elem).html())
  $(elem).prop("disabled", true)
  $(elem).html('<i class="spinner-border spinner-border-sm"></i>  ...')
}

function BtnReset(elem) {
  $(elem).prop("disabled", false)
  $(elem).html($(elem).attr("data-original-text"))
}
