<template>
  <div class="card">
    <ul class="list-group list-group-flush" id="addedTransducers"></ul>
  </div>
  <br />
  <div id="newTransducerButton">
    <fieldset>
      <button class="btn btn-primary" type="button" onclick="newTransducer()">
        Add Transducer
      </button>
    </fieldset>
  </div>
  <form id="createTransducerForm" hidden>
    <fieldset>
      <div class="mb-3">
        <label for="transducerType" class="form-label">Source Transducer</label>
        <select class="form-select" id="transducerType" name="transducerType" onchange="transducerSelected(this)"
          aria-label="Select a transducer">
          <option value selected>-- Select transducer --</option>
          <!-- <option value="{{transducerType}}">{{ transducerTypeLabel }}</option> -->
        </select>
      </div>
      <div id="transducerSettings"></div>
      <div id="formControls" hidden>
        <button type="button" class="btn btn-primary btn-block" onclick="cancelAddTransducer(this)">
          Cancel
        </button>
        <button type="button" id="addTransducerButton" class="btn btn-primary btn-block" onclick="addTransducer(this)">
          Add
        </button>
        <button type="button" data-transducer-id="-1" id="saveTransducerButton" class="btn btn-primary btn-block"
          onclick="saveTransducer(this, this.dataset.transducerId)" hidden>
          Save Transducer
        </button>
      </div>
    </fieldset>
  </form>
</template>

<script>
export default {
}
</script>

<style scoped></style>
