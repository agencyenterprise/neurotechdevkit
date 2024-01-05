<template>
  <div class="container-fluid">
    <aside class="parameters">
      <Parameters />
    </aside>
    <div class="main-grid">
      <MainGrid />
    </div>
  </div>
</template>

<script>
import Parameters from './components/Parameters.vue'
import MainGrid from './components/MainGrid.vue'


export default {
  name: 'App',
  components: {
    Parameters,
    MainGrid
  }
}
</script>

<style>
.container-fluid {
  height: 100%;
  width: 100%;
  margin: 0;
  padding: 0;
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
  /* background-color: #fafafa; */
  margin-left: 450px;
  height: 100%;
  overflow: hidden;
  padding: 10px;
}
</style>
