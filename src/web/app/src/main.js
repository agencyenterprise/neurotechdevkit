import { createApp } from "vue";
import App from "./App.vue";
import { library } from "@fortawesome/fontawesome-svg-core";
import { faChevronUp, faChevronDown } from "@fortawesome/free-solid-svg-icons";
import { FontAwesomeIcon } from "@fortawesome/vue-fontawesome";
import "bootstrap/dist/css/bootstrap.css";
import "bootstrap/dist/js/bootstrap.js";
import store from "./store";

library.add(faChevronDown, faChevronUp);

const app = createApp(App);

app.use(store);
app.component("font-awesome-icon", FontAwesomeIcon);
app.mount("#app");
