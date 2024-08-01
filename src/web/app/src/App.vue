<template>
  <div class="container-fluid">
    <aside class="parameters">
      <ParametersPanel />
    </aside>
    <div class="main-grid">
      <MainGrid />
    </div>
  </div>
</template>

<script>
import ParametersPanel from './components/ParametersPanel.vue'
import MainGrid from './components/MainGrid.vue'

export default {
  name: 'App',
  components: {
    ParametersPanel,
    MainGrid
  }
}
</script>

<style>
.container-fluid {
  height: 100%;
  width: 100%;
  margin: 0px;
  padding-left: 0 !important;
}

.parameters {
  background-color: #e4e5f1;
  position: absolute;
  width: 450px;
  height: 100%;
  padding: 10px;
  overflow: scroll;
}

.main-grid {
  margin-left: 450px;
  height: 100%;
  overflow: hidden;
}
</style>
